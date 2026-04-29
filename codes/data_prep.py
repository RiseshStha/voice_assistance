"""
Data preprocessing, verification, EDA, and SMOTE-based dataset balancing.
"""

from __future__ import annotations

import io
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

from config import (
    BALANCED_QA_EXCEL,
    DATA_STATUS_JSON,
    INTENTS,
    OUTPUTS_DIR,
    QA_EXCEL,
    ROOT,
    pick_raw_qa_excel,
)

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

RAW_DATA = pick_raw_qa_excel()
NEPALI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
TEXT_COLUMNS = ["question", "answer", "intent", "subject"]


def normalize_nepali(text: object) -> str:
    text = "" if pd.isna(text) else str(text)
    text = text.strip().translate(NEPALI_DIGITS)
    text = re.sub(r"^[\d]+[\.\)\s।]+", "", text).strip()
    noise_patterns = [
        r"\bलेख्नुहोस्\b",
        r"\bबताउनुहोस्\b",
        r"\bउदाहरण दिनुहोस्\b",
        r"\bउल्लेख गर्नुहोस्\b",
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" ?।.!," )


def normalize_generic_text(text: object) -> str:
    text = "" if pd.isna(text) else str(text)
    text = text.translate(NEPALI_DIGITS)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_grade(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = normalize_generic_text(value)
    if not text:
        return pd.NA
    if text.isdigit():
        return int(text)
    return pd.NA


def fill_with_group_mode(df: pd.DataFrame, target: str, groups: list[list[str]]) -> pd.Series:
    series = df[target].copy()
    for keys in groups:
        group_mode = df.groupby(keys, dropna=False)[target].transform(
            lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else pd.NA
        )
        series = series.where(series.notna(), group_mode).infer_objects(copy=False)
    fallback_mode = series.dropna().mode()
    if not fallback_mode.empty:
        series = series.fillna(fallback_mode.iloc[0]).infer_objects(copy=False)
    return series


def prepare_clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    original_columns = list(df.columns)
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = ["question", "answer", "intent"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    for column in TEXT_COLUMNS:
        if column in df.columns:
            df[column] = df[column].map(normalize_generic_text)

    df["question_original"] = df["question"]
    df["question"] = df["question"].map(normalize_nepali)
    df["answer"] = df["answer"].map(normalize_generic_text)
    df["intent"] = df["intent"].map(normalize_generic_text).str.lower()

    if "subject" not in df.columns:
        df["subject"] = "unknown"
    if "grade" not in df.columns:
        df["grade"] = pd.NA

    blank_subjects = df["subject"].eq("")
    df.loc[blank_subjects, "subject"] = pd.NA

    df["grade"] = df["grade"].map(normalize_grade)
    df["grade"] = fill_with_group_mode(df, "grade", [["subject", "intent"], ["intent"], ["subject"]])

    blank_questions = df["question"].eq("")
    if blank_questions.any():
        df.loc[blank_questions, "question"] = df.loc[blank_questions, "question_original"].map(normalize_generic_text)

    blank_answers = df["answer"].eq("")
    if blank_answers.any():
        df.loc[blank_answers, "answer"] = "उत्तर उपलब्ध छैन"

    blank_subjects = df["subject"].isna() | df["subject"].eq("")
    if blank_subjects.any():
        df["subject"] = fill_with_group_mode(df.assign(subject=df["subject"]), "subject", [["intent"]])
        df["subject"] = df["subject"].fillna("unknown")

    blank_intents = df["intent"].eq("")
    if blank_intents.any():
        intent_mode = df.loc[~blank_intents, "intent"].mode()
        df.loc[blank_intents, "intent"] = intent_mode.iloc[0] if not intent_mode.empty else "unknown_intent"

    df["grade"] = df["grade"].astype("Int64")
    df["source_row_id"] = range(1, len(df) + 1)
    df["data_origin"] = "original"

    stats = {
        "raw_rows": int(len(df)),
        "raw_columns": original_columns,
        "duplicate_rows": int(df.duplicated(subset=["question", "answer", "intent", "subject", "grade"]).sum()),
        "duplicate_questions": int(df.duplicated(subset=["question"]).sum()),
        "blank_questions_after_cleaning": int(df["question"].eq("").sum()),
        "blank_answers_after_cleaning": int(df["answer"].eq("").sum()),
        "missing_grades_after_cleaning": int(df["grade"].isna().sum()),
    }
    return df, stats


def summarize_class_coverage(df: pd.DataFrame) -> dict:
    present_intents = sorted(df["intent"].astype(str).unique().tolist())
    expected_intents = sorted(INTENTS)
    missing_intents = [intent for intent in expected_intents if intent not in present_intents]
    unexpected_intents = [intent for intent in present_intents if intent not in expected_intents]
    return {
        "expected_intents": expected_intents,
        "present_intents": present_intents,
        "missing_intents": missing_intents,
        "unexpected_intents": unexpected_intents,
        "all_expected_classes_available": len(missing_intents) == 0,
        "present_class_count": len(present_intents),
    }


def build_smote_balanced_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    counts_before = df["intent"].value_counts().sort_index()
    class_count = len(counts_before)
    target_per_class = max(int(counts_before.max()), math.ceil(4000 / class_count))

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        sublinear_tf=True,
        min_df=1,
    )
    X_text = vectorizer.fit_transform(df["question"].astype(str))

    max_components = min(300, X_text.shape[0] - 1, X_text.shape[1] - 1)
    n_components = max(2, max_components)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X_text)

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["intent"].astype(str))
    class_counter = Counter(y)
    min_class_size = min(class_counter.values())
    k_neighbors = max(1, min(5, min_class_size - 1))

    sampling_strategy = {
        label: target_per_class for label, count in class_counter.items() if count < target_per_class
    }
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X_reduced, y)

    original_count = len(df)
    synthetic_vectors = X_resampled[original_count:]
    synthetic_labels = y_resampled[original_count:]

    synthetic_rows = []
    for synthetic_idx, (vector, label) in enumerate(zip(synthetic_vectors, synthetic_labels), start=1):
        class_name = encoder.inverse_transform([label])[0]
        class_mask = df["intent"].eq(class_name).to_numpy()
        class_vectors = X_reduced[class_mask]
        class_rows = df.loc[class_mask].reset_index(drop=True)

        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(class_vectors)
        distance, neighbor_index = nn.kneighbors([vector], n_neighbors=1)
        anchor_row = class_rows.iloc[int(neighbor_index[0][0])].copy()
        anchor_row["data_origin"] = "smote_generated"
        anchor_row["smote_reference_row_id"] = int(anchor_row["source_row_id"])
        anchor_row["smote_distance"] = float(distance[0][0])
        anchor_row["source_row_id"] = original_count + synthetic_idx
        synthetic_rows.append(anchor_row)

    synthetic_df = pd.DataFrame(synthetic_rows)
    if synthetic_df.empty:
        balanced_df = df.copy()
    else:
        balanced_df = pd.concat([df, synthetic_df], ignore_index=True)

    counts_after = balanced_df["intent"].value_counts().sort_index()
    summary = {
        "target_per_class": int(target_per_class),
        "k_neighbors": int(k_neighbors),
        "total_rows_after_smote": int(len(balanced_df)),
        "synthetic_rows_added": int(len(balanced_df) - len(df)),
        "intent_counts_before": {k: int(v) for k, v in counts_before.items()},
        "intent_counts_after": {k: int(v) for k, v in counts_after.items()},
        "origin_counts": {
            k: int(v) for k, v in balanced_df["data_origin"].value_counts().sort_index().items()
        },
    }
    return balanced_df, summary


def plot_intent_distribution(counts: pd.Series, output_path: Path, title: str, color: str) -> None:
    plt.figure(figsize=(11, 6))
    ax = sns.barplot(x=counts.index, y=counts.values, color=color)
    ax.set_title(title)
    ax.set_xlabel("Intent")
    ax.set_ylabel("Samples")
    plt.xticks(rotation=30, ha="right")
    for idx, value in enumerate(counts.values):
        ax.text(idx, value + max(counts.values) * 0.01, str(int(value)), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_status_dashboard(before: pd.Series, after: pd.Series, origin_counts: pd.Series, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.barplot(x=before.index, y=before.values, ax=axes[0], color="#4C78A8")
    axes[0].set_title("Before SMOTE")
    axes[0].set_xlabel("Intent")
    axes[0].set_ylabel("Samples")
    axes[0].tick_params(axis="x", rotation=30)

    sns.barplot(x=after.index, y=after.values, ax=axes[1], color="#F58518")
    axes[1].set_title("After SMOTE")
    axes[1].set_xlabel("Intent")
    axes[1].set_ylabel("Samples")
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].pie(origin_counts.values, labels=origin_counts.index, autopct="%1.1f%%", startangle=90)
    axes[2].set_title("Original vs SMOTE-Generated")

    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    print(f"Loading dataset from {RAW_DATA}...")
    raw_df = pd.read_excel(RAW_DATA)

    clean_df, clean_stats = prepare_clean_dataframe(raw_df)
    class_coverage = summarize_class_coverage(clean_df)
    balanced_df, smote_stats = build_smote_balanced_dataframe(clean_df)

    before_counts = clean_df["intent"].value_counts().sort_index()
    after_counts = balanced_df["intent"].value_counts().sort_index()
    origin_counts = balanced_df["data_origin"].value_counts().sort_index()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_intent_distribution(
        before_counts,
        OUTPUTS_DIR / "intent_distribution_before_smote.png",
        "Intent Distribution Before SMOTE",
        "#4C78A8",
    )
    plot_intent_distribution(
        after_counts,
        OUTPUTS_DIR / "intent_distribution_after_smote.png",
        "Intent Distribution After SMOTE",
        "#F58518",
    )
    plot_status_dashboard(
        before_counts,
        after_counts,
        origin_counts,
        OUTPUTS_DIR / "data_status_dashboard.png",
    )

    clean_df.to_excel(QA_EXCEL, index=False)
    balanced_df.to_excel(BALANCED_QA_EXCEL, index=False)

    summary = {
        "raw_file": str(RAW_DATA),
        "clean_output": str(QA_EXCEL),
        "balanced_output": str(BALANCED_QA_EXCEL),
        "cleaning": clean_stats,
        "class_coverage": class_coverage,
        "smote": smote_stats,
    }
    with open(DATA_STATUS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Verified rows: {clean_stats['raw_rows']}")
    print(f"Duplicate full rows retained: {clean_stats['duplicate_rows']}")
    print(f"Duplicate questions retained: {clean_stats['duplicate_questions']}")
    print(f"Missing grades after cleaning: {clean_stats['missing_grades_after_cleaning']}")
    print(f"Expected classes available: {class_coverage['all_expected_classes_available']}")
    print(f"Present intents: {', '.join(class_coverage['present_intents'])}")
    if class_coverage["missing_intents"]:
        print(f"Missing intents: {', '.join(class_coverage['missing_intents'])}")
    print(f"Balanced dataset size: {smote_stats['total_rows_after_smote']}")
    print(f"Per-class target after SMOTE: {smote_stats['target_per_class']}")
    print(f"Clean dataset saved to: {QA_EXCEL}")
    print(f"SMOTE-balanced dataset saved to: {BALANCED_QA_EXCEL}")
    print(f"Plots saved in: {OUTPUTS_DIR}")
    print(f"Status summary saved to: {DATA_STATUS_JSON}")


if __name__ == "__main__":
    main()
