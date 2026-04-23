"""
evaluate_and_benchmark.py
=========================
Reproducible evaluation + visualization + latency benchmarks for marking.

Generates (in outputs/):
- eval_report.json / eval_report.txt
- confusion_matrix_eval.png
- accuracy_table.png
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

import pickle

from config import CLEAN_XL, MODEL_PATH, KB_INDEX_PATH, OUTPUTS_DIR


@dataclass
class LatStats:
    mean: float
    median: float
    p95: float
    max: float
    min: float


def _ms_stats(samples_ms: list[float]) -> LatStats:
    arr = np.asarray(samples_ms, dtype=float)
    if arr.size == 0:
        return LatStats(mean=0.0, median=0.0, p95=0.0, max=0.0, min=0.0)
    return LatStats(
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        p95=float(np.percentile(arr, 95)),
        max=float(np.max(arr)),
        min=float(np.min(arr)),
    )


def _load_model():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run: python .\\codes\\train_model.py"
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _load_kb():
    """
    Load retrieval KB if present; otherwise build quickly from CLEAN_XL.
    Mirrors the logic used in app.py so latency numbers match the demo.
    """
    try:
        if Path(KB_INDEX_PATH).exists():
            with open(KB_INDEX_PATH, "rb") as f:
                kb = pickle.load(f)
            required = {"vectorizer", "X", "questions", "answers", "intents"}
            if required.issubset(set(kb.keys())) and len(kb["questions"]) > 0:
                return kb
    except Exception:
        pass

    from sklearn.feature_extraction.text import TfidfVectorizer

    df = pd.read_excel(CLEAN_XL).dropna(subset=["question", "answer", "intent"])
    questions = df["question"].astype(str).tolist()
    answers = df["answer"].astype(str).tolist()
    intents = df["intent"].astype(str).tolist()

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        sublinear_tf=True,
        min_df=2,
    )
    X = vectorizer.fit_transform(questions)
    return {"vectorizer": vectorizer, "X": X, "questions": questions, "answers": answers, "intents": intents}


def _retrieve_answer(kb: dict, user_text: str, predicted_intent: str, min_sim: float = 0.25):
    user_text = (user_text or "").strip()
    if not user_text:
        return None, 0.0, None

    vec = kb["vectorizer"]
    X = kb["X"]
    q_vec = vec.transform([user_text])

    intents = kb.get("intents")
    mask_idx = None
    if intents is not None and predicted_intent:
        mask_idx = [i for i, it in enumerate(intents) if it == predicted_intent]

    if mask_idx:
        X_sub = X[mask_idx]
        sims_sub = (X_sub @ q_vec.T).toarray().reshape(-1)
        best_local = int(np.argmax(sims_sub))
        best_idx = mask_idx[best_local]
        best_sim = float(sims_sub[best_local])
    else:
        sims = (X @ q_vec.T).toarray().reshape(-1)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

    if best_sim < min_sim:
        return None, best_sim, kb["questions"][best_idx]
    return kb["answers"][best_idx], best_sim, kb["questions"][best_idx]


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load data (same split as training) ----
    df = pd.read_excel(CLEAN_XL)
    df = df.dropna(subset=["question", "intent"])
    X = df["question"].astype(str).values
    y = df["intent"].astype(str).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- Load model + evaluate ----
    model = _load_model()
    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))

    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    labels = sorted(np.unique(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()

    # ---- Confusion matrix heatmap (visualization) ----
    cm_png = OUTPUTS_DIR / "confusion_matrix_eval.png"
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        np.asarray(cm),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Confusion Matrix — Intent Classification (SVM)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(cm_png, dpi=220)
    plt.close()

    # ---- Accuracy results table (visualization) ----
    table_png = OUTPUTS_DIR / "accuracy_table.png"
    metrics_rows = [
        ["SVM (LinearSVC)", f"{acc*100:.2f}%", f"{macro_f1*100:.2f}%", f"{weighted_f1*100:.2f}%"],
    ]
    col_labels = ["Model", "Accuracy", "Macro F1", "Weighted F1"]

    fig, ax = plt.subplots(figsize=(9.2, 2.2))
    ax.axis("off")
    tbl = ax.table(
        cellText=metrics_rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)
    plt.title("Accuracy Results Table (Test Split)")
    plt.tight_layout()
    plt.savefig(table_png, dpi=220)
    plt.close(fig)

    # ---- Latency benchmarks (intent + retrieval + total) ----
    kb = _load_kb()
    nq = min(50, len(X_test))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), size=nq, replace=False)
    queries = X_test[idx].tolist()

    intent_ms: list[float] = []
    retrieval_ms: list[float] = []
    total_ms: list[float] = []

    for q in queries:
        t0 = time.perf_counter()
        t_int0 = time.perf_counter()
        intent = model.predict([q])[0]
        t_int1 = time.perf_counter()

        t_ret0 = time.perf_counter()
        _retrieve_answer(kb, q, intent)
        t_ret1 = time.perf_counter()
        t1 = time.perf_counter()

        intent_ms.append((t_int1 - t_int0) * 1000.0)
        retrieval_ms.append((t_ret1 - t_ret0) * 1000.0)
        total_ms.append((t1 - t0) * 1000.0)

    intent_stats = _ms_stats(intent_ms)
    retrieval_stats = _ms_stats(retrieval_ms)
    total_stats = _ms_stats(total_ms)

    out = {
        "intent_accuracy": {
            "test_samples": int(len(X_test)),
            "accuracy": acc,
            "accuracy_pct": round(acc * 100.0, 2),
            "macro_f1": round(macro_f1, 6),
            "weighted_f1": round(weighted_f1, 6),
            "report": report_dict,
            "confusion_matrix": cm,
            "labels": labels,
            "artifacts": {
                "confusion_matrix_heatmap_png": str(cm_png),
                "accuracy_table_png": str(table_png),
            },
        },
        "latency": {
            "n_queries": int(nq),
            "with_tts": False,
            "intent": intent_stats.__dict__,
            "retrieval": retrieval_stats.__dict__,
            "total": total_stats.__dict__,
        },
    }

    # ---- Write reports ----
    json_path = OUTPUTS_DIR / "eval_report.json"
    txt_path = OUTPUTS_DIR / "eval_report.txt"

    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    txt = []
    txt.append("NEPALI STUDENT VOICE ASSISTANT — EVALUATION REPORT")
    txt.append("=" * 60)
    txt.append("")
    txt.append("[A] INTENT CLASSIFICATION (SVM)")
    txt.append(f"  Test Accuracy  : {acc*100:.2f}%")
    txt.append(f"  Macro F1       : {macro_f1*100:.2f}%")
    txt.append(f"  Weighted F1    : {weighted_f1*100:.2f}%")
    txt.append("")
    txt.append("[B] LATENCY (ms) — per query (no TTS)")
    txt.append(f"  Intent   mean/p95 : {intent_stats.mean:.1f} / {intent_stats.p95:.1f} ms")
    txt.append(f"  Retrieval mean/p95: {retrieval_stats.mean:.1f} / {retrieval_stats.p95:.1f} ms")
    txt.append(f"  Total mean/p95    : {total_stats.mean:.1f} / {total_stats.p95:.1f} ms")
    txt.append("")
    txt.append("[C] VISUALIZATIONS")
    txt.append(f"  Confusion matrix heatmap: {cm_png}")
    txt.append(f"  Accuracy results table  : {table_png}")
    txt.append("")

    txt_path.write_text("\n".join(txt), encoding="utf-8")

    print("Saved:")
    print(" -", json_path)
    print(" -", txt_path)
    print(" -", cm_png)
    print(" -", table_png)


if __name__ == "__main__":
    main()

