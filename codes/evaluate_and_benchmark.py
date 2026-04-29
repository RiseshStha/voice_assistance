"""
evaluate_and_benchmark.py
=========================
Reproducible evaluation + visualization + latency benchmarks for marking.
Evaluates SVM against comparative baseline models.
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

def _load_kb():
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

def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    df = pd.read_excel(CLEAN_XL)
    df = df.dropna(subset=["question", "intent"])
    X = df["question"].astype(str).values
    y = df["intent"].astype(str).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models_to_eval = {
        "Linear SVM (Proposed)": MODEL_PATH,
        "Logistic Regression": OUTPUTS_DIR / "model_lr.pkl",
        "KNN (k=7, distance)": OUTPUTS_DIR / "model_knn.pkl",
    }

    metrics_rows = []
    full_report = {
        "intent_accuracy": None,
        "latency": None,
    }
    
    # Preload KB for latency benchmark
    kb = _load_kb()
    nq = min(50, len(X_test))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), size=nq, replace=False)
    queries = X_test[idx].tolist()

    for name, path in models_to_eval.items():
        if not path.exists():
            print(f"Skipping {name}, not found at {path}")
            continue
            
        with open(path, "rb") as f:
            model = pickle.load(f)
            
        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
        weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
        
        # Latency Benchmark
        intent_ms = []
        for q in queries:
            t0 = time.perf_counter()
            _ = model.predict([q])[0]
            t1 = time.perf_counter()
            intent_ms.append((t1 - t0) * 1000.0)
            
        lat_stats = _ms_stats(intent_ms)
        
        metrics_rows.append([
            name, 
            f"{acc*100:.2f}%", 
            f"{macro_f1*100:.2f}%", 
            f"{weighted_f1*100:.2f}%",
            f"{lat_stats.mean:.1f} ms"
        ])

        if name == "Linear SVM (Proposed)":
            labels = sorted(np.unique(y))
            cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()
            cm_png = OUTPUTS_DIR / "confusion_matrix_eval.png"
            report_json = OUTPUTS_DIR / "classification_report_eval.json"
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
            with open(report_json, "w", encoding="utf-8") as f:
                clf_report = classification_report(
                    y_test, y_pred, labels=labels, output_dict=True, zero_division=0
                )
                json.dump(clf_report, f, ensure_ascii=False, indent=2)

            full_report["intent_accuracy"] = {
                "test_samples": int(len(y_test)),
                "accuracy": acc,
                "accuracy_pct": round(acc * 100, 2),
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "report": clf_report,
                "confusion_matrix": cm,
                "labels": labels,
                "artifacts": {
                    "confusion_matrix_heatmap_png": str(cm_png),
                    "accuracy_table_png": str(OUTPUTS_DIR / "accuracy_table.png"),
                },
            }
            full_report["latency"] = {
                "n_queries": int(nq),
                "with_tts": False,
                "intent": lat_stats.__dict__,
            }

    # ---- Accuracy results table (visualization) ----
    table_png = OUTPUTS_DIR / "accuracy_table.png"
    col_labels = ["Model", "Accuracy", "Macro F1", "Weighted F1", "Inference Latency"]

    fig, ax = plt.subplots(figsize=(10, 2.5))
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
    
    # Make SVM bold if present in the table
    for row_idx, row in enumerate(metrics_rows, start=1):
        if row[0] == "Linear SVM (Proposed)":
            for col_idx in range(len(col_labels)):
                tbl[(row_idx, col_idx)].get_text().set_weight("bold")
            break
        
    plt.title("Model Comparison Table (Test Split)")
    plt.tight_layout()
    plt.savefig(table_png, dpi=220)
    plt.close(fig)

    if full_report["intent_accuracy"] is not None:
        # Retrieval/total latency are only meaningful for the proposed SVM pipeline report.
        kb_vectorizer = kb["vectorizer"]
        kb_X = kb["X"]
        kb_intents = np.asarray(kb["intents"])
        retrieval_ms = []
        total_ms = []

        with open(MODEL_PATH, "rb") as f:
            svm_model = pickle.load(f)

        for q in queries:
            t0 = time.perf_counter()
            pred = svm_model.predict([q])[0]
            t1 = time.perf_counter()

            qv = kb_vectorizer.transform([q])
            mask = kb_intents == pred
            if np.any(mask):
                _ = (qv @ kb_X[mask].T).toarray()
            t2 = time.perf_counter()

            retrieval_ms.append((t2 - t1) * 1000.0)
            total_ms.append((t2 - t0) * 1000.0)

        retrieval_stats = _ms_stats(retrieval_ms)
        total_stats = _ms_stats(total_ms)
        full_report["latency"]["retrieval"] = retrieval_stats.__dict__
        full_report["latency"]["total"] = total_stats.__dict__

        eval_json = OUTPUTS_DIR / "eval_report.json"
        eval_txt = OUTPUTS_DIR / "eval_report.txt"
        with open(eval_json, "w", encoding="utf-8") as f:
            json.dump(full_report, f, ensure_ascii=False, indent=2)

        # ---- Latency Bar Chart (visualization) ----
        latency_png = OUTPUTS_DIR / "fig_latency.png"
        stages = ["Intent Classification", "KB Retrieval", "Total Pipeline"]
        means = [full_report['latency']['intent']['mean'], full_report['latency']['retrieval']['mean'], full_report['latency']['total']['mean']]
        p95s = [full_report['latency']['intent']['p95'], full_report['latency']['retrieval']['p95'], full_report['latency']['total']['p95']]

        x = np.arange(len(stages))
        width = 0.35

        fig_lat, ax_lat = plt.subplots(figsize=(8, 5))
        rects1 = ax_lat.bar(x - width/2, means, width, label='Mean Latency', color='skyblue')
        rects2 = ax_lat.bar(x + width/2, p95s, width, label='P95 Latency', color='salmon')

        ax_lat.set_ylabel('Time (ms)')
        ax_lat.set_title('Pipeline Latency Benchmarks (CPU)')
        ax_lat.set_xticks(x)
        ax_lat.set_xticklabels(stages)
        ax_lat.legend()

        ax_lat.bar_label(rects1, fmt='%.1f', padding=3)
        ax_lat.bar_label(rects2, fmt='%.1f', padding=3)

        fig_lat.tight_layout()
        fig_lat.savefig(latency_png, dpi=200)
        plt.close(fig_lat)

        summary = (
            "NEPALI STUDENT VOICE ASSISTANT - EVALUATION REPORT\n"
            "============================================================\n\n"
            "[A] INTENT CLASSIFICATION (SVM)\n"
            f"  Test Accuracy  : {full_report['intent_accuracy']['accuracy_pct']:.2f}%\n"
            f"  Macro F1       : {full_report['intent_accuracy']['macro_f1'] * 100:.2f}%\n"
            f"  Weighted F1    : {full_report['intent_accuracy']['weighted_f1'] * 100:.2f}%\n\n"
            "[B] LATENCY (ms) - per query (no TTS)\n"
            f"  Intent   mean/p95 : {full_report['latency']['intent']['mean']:.1f} / {full_report['latency']['intent']['p95']:.1f} ms\n"
            f"  Retrieval mean/p95: {full_report['latency']['retrieval']['mean']:.1f} / {full_report['latency']['retrieval']['p95']:.1f} ms\n"
            f"  Total mean/p95    : {full_report['latency']['total']['mean']:.1f} / {full_report['latency']['total']['p95']:.1f} ms\n\n"
            "[C] VISUALIZATIONS\n"
            f"  Confusion matrix heatmap: {OUTPUTS_DIR / 'confusion_matrix_eval.png'}\n"
            f"  Accuracy results table  : {table_png}\n"
            f"  Latency bar chart       : {latency_png}\n"
        )
        with open(eval_txt, "w", encoding="utf-8") as f:
            f.write(summary)

    print("Evaluation Complete. Comparison table saved to", table_png)

if __name__ == "__main__":
    main()

