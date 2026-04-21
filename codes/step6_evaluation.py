"""
STEP 6 — Evaluation & Benchmarking
=====================================
Measures the full system on three axes:

  A. Intent Classification accuracy  (already in step2_intent.py, summarised here)
  B. End-to-end pipeline latency     (intent + retrieval + optional TTS)
  C. Resource usage                  (RAM & CPU via psutil)

What it produces
----------------
  outputs/eval_report.json     — machine-readable full report
  outputs/eval_report.txt      — human-readable summary table

Usage
-----
  # Full evaluation (requires trained model + built KB index)
  python step6_evaluation.py

  # Quick latency-only benchmark (N text queries, no TTS)
  python step6_evaluation.py --bench-n 50

  # Include TTS latency
  python step6_evaluation.py --with-tts

  # Monitor RAM/CPU live for 30 s while running queries
  python step6_evaluation.py --resource-monitor
"""

import sys, json, time, logging, argparse, gc
from pathlib import Path

import numpy  as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import QA_EXCEL, OUTPUTS_DIR, SVM_MODEL_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 6.0  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ram_mb() -> float:
    """Return current process RSS in MB."""
    try:
        import psutil, os
        return psutil.Process(os.getpid()).memory_info().rss / 1_048_576
    except ImportError:
        return -1.0


def _cpu_percent(interval: float = 0.5) -> float:
    """Return process CPU usage %."""
    try:
        import psutil, os
        return psutil.Process(os.getpid()).cpu_percent(interval=interval)
    except ImportError:
        return -1.0


# ─────────────────────────────────────────────────────────────────────────────
# 6.1  Intent classification evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_intent(n_samples: int = None) -> dict:
    """
    Re-evaluate the intent classifier on a held-out test set.
    Returns accuracy and per-class metrics.
    """
    from step2_intent import load_dataset, load_pipeline, build_pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, classification_report,
                                  confusion_matrix)

    df = load_dataset()
    X  = df["question"].tolist()
    y  = df["intent"].tolist()

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    if n_samples:
        X_test = X_test[:n_samples]
        y_test = y_test[:n_samples]

    pipe   = load_pipeline()
    y_pred = pipe.predict(X_test)

    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm     = confusion_matrix(y_test, y_pred, labels=list(pipe.classes_)).tolist()

    log.info(f"\n{classification_report(y_test, y_pred, zero_division=0)}")
    log.info(f"Intent Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    return {
        "test_samples":   len(y_test),
        "accuracy":       round(acc, 6),
        "accuracy_pct":   round(acc * 100, 2),
        "report":         report,
        "confusion_matrix": cm,
        "labels":         list(pipe.classes_),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.2  End-to-end latency benchmark (text queries only)
# ─────────────────────────────────────────────────────────────────────────────

def eval_latency(
    svm_pipe,
    kb_index,
    n:        int  = 50,
    with_tts: bool = False,
) -> dict:
    """
    Measure per-stage latency over N random questions from the dataset.

    Returns dict with mean/median/p95/max for each stage.
    """
    df      = pd.read_excel(QA_EXCEL, dtype=str).dropna(subset=["question"])
    samples = df["question"].sample(min(n, len(df)), random_state=7).tolist()

    from step2_intent   import predict_intent
    from step3_retrieval import retrieve
    if with_tts:
        from step4_tts import speak

    stage_times = {"intent_ms": [], "retrieval_ms": [], "tts_ms": [], "total_ms": []}

    log.info(f"Running latency benchmark: {len(samples)} queries, TTS={with_tts} …")
    for q in samples:
        t_total = time.time()

        # Intent
        t0  = time.time()
        res = predict_intent(q, svm_pipe)
        stage_times["intent_ms"].append((time.time() - t0) * 1000)

        # Retrieval
        t0  = time.time()
        hits = retrieve(q, kb_index, top_n=1, intent=res["intent"])
        stage_times["retrieval_ms"].append((time.time() - t0) * 1000)

        # TTS
        t0 = time.time()
        if with_tts and hits:
            speak(hits[0]["answer"])
        stage_times["tts_ms"].append((time.time() - t0) * 1000)

        stage_times["total_ms"].append((time.time() - t_total) * 1000)

    def _stats(vals):
        a = np.array(vals)
        return {
            "mean":   round(float(a.mean()),              1),
            "median": round(float(np.median(a)),          1),
            "p95":    round(float(np.percentile(a, 95)),  1),
            "max":    round(float(a.max()),                1),
            "min":    round(float(a.min()),                1),
        }

    results = {
        "n_queries":   len(samples),
        "with_tts":    with_tts,
        "intent":      _stats(stage_times["intent_ms"]),
        "retrieval":   _stats(stage_times["retrieval_ms"]),
        "total":       _stats(stage_times["total_ms"]),
    }
    if with_tts:
        results["tts"] = _stats(stage_times["tts_ms"])

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6.3  Resource monitoring
# ─────────────────────────────────────────────────────────────────────────────

def eval_resources(svm_pipe, kb_index, duration_s: int = 30) -> dict:
    """
    Monitor RAM and CPU while running random queries for `duration_s` seconds.
    Requires: pip install psutil
    """
    try:
        import psutil, os
    except ImportError:
        log.warning("psutil not installed. Run: pip install psutil")
        return {"error": "psutil not installed"}

    df      = pd.read_excel(QA_EXCEL, dtype=str).dropna(subset=["question"])
    samples = df["question"].tolist()

    from step2_intent    import predict_intent
    from step3_retrieval import retrieve

    proc     = psutil.Process(os.getpid())
    ram_vals = []
    cpu_vals = []
    t_end    = time.time() + duration_s
    q_count  = 0

    log.info(f"Resource monitor: running queries for {duration_s}s …")
    while time.time() < t_end:
        q = samples[q_count % len(samples)]
        res = predict_intent(q, svm_pipe)
        retrieve(q, kb_index, top_n=1, intent=res["intent"])
        q_count += 1

        ram_vals.append(proc.memory_info().rss / 1_048_576)
        cpu_vals.append(proc.cpu_percent(interval=None))

    def _stats(vals):
        a = np.array(vals)
        return {
            "mean": round(float(a.mean()), 1),
            "max":  round(float(a.max()),  1),
            "min":  round(float(a.min()),  1),
        }

    results = {
        "duration_s":  duration_s,
        "queries_run": q_count,
        "qps":         round(q_count / duration_s, 1),
        "ram_mb":      _stats(ram_vals),
        "cpu_pct":     _stats(cpu_vals),
    }
    log.info(f"RAM  — mean={results['ram_mb']['mean']} MB  max={results['ram_mb']['max']} MB")
    log.info(f"CPU  — mean={results['cpu_pct']['mean']}%  max={results['cpu_pct']['max']}%")
    log.info(f"QPS  — {results['qps']} queries/s")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6.4  Print & save report
# ─────────────────────────────────────────────────────────────────────────────

def print_report(intent_res: dict, latency_res: dict, resource_res: dict = None):
    sep = "═" * 65
    print(f"\n{sep}")
    print("  FULL SYSTEM EVALUATION REPORT")
    print(sep)

    # Intent
    print("\n  [A] INTENT CLASSIFICATION")
    print(f"      Samples tested : {intent_res.get('test_samples', 'N/A')}")
    print(f"      Test Accuracy  : {intent_res.get('accuracy_pct', 'N/A'):.2f}%")
    rep = intent_res.get("report", {})
    print(f"      Macro F1       : {rep.get('macro avg', {}).get('f1-score', 0)*100:.2f}%")
    print(f"      Weighted F1    : {rep.get('weighted avg', {}).get('f1-score', 0)*100:.2f}%")

    # Latency
    print("\n  [B] END-TO-END LATENCY  (intent + retrieval, no ASR/TTS)")
    t = latency_res
    print(f"      Queries        : {t.get('n_queries')}")
    print(f"      Intent  mean   : {t.get('intent', {}).get('mean', 'N/A')} ms")
    print(f"      Retrieval mean : {t.get('retrieval', {}).get('mean', 'N/A')} ms")
    print(f"      Total   mean   : {t.get('total', {}).get('mean', 'N/A')} ms  "
          f" | p95={t.get('total', {}).get('p95', 'N/A')} ms  | max={t.get('total', {}).get('max', 'N/A')} ms")

    # Resources
    if resource_res and "error" not in resource_res:
        r = resource_res
        print("\n  [C] RESOURCE USAGE")
        print(f"      Queries/s      : {r['qps']}")
        print(f"      RAM  mean/max  : {r['ram_mb']['mean']} / {r['ram_mb']['max']} MB")
        print(f"      CPU  mean/max  : {r['cpu_pct']['mean']} / {r['cpu_pct']['max']} %")

    print(f"\n{sep}\n")


def save_report(report: dict):
    json_path = OUTPUTS_DIR / "eval_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log.info(f"JSON report saved → {json_path}")

    txt_path = OUTPUTS_DIR / "eval_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        def _w(s=""): f.write(s + "\n")
        _w("NEPALI STUDENT VOICE ASSISTANT — EVALUATION REPORT")
        _w("=" * 60)

        ia = report.get("intent_accuracy", {})
        _w("\n[A] INTENT CLASSIFICATION")
        _w(f"  Test Accuracy  : {ia.get('accuracy_pct', 'N/A'):.2f}%")
        rep = ia.get("report", {})
        _w(f"  Macro F1       : {rep.get('macro avg', {}).get('f1-score', 0)*100:.2f}%")
        _w(f"  Weighted F1    : {rep.get('weighted avg', {}).get('f1-score', 0)*100:.2f}%")

        la = report.get("latency", {})
        _w("\n[B] LATENCY (ms)")
        _w(f"  Intent   mean  : {la.get('intent',{}).get('mean','?')} ms")
        _w(f"  Retrieval mean : {la.get('retrieval',{}).get('mean','?')} ms")
        _w(f"  Total mean/p95 : {la.get('total',{}).get('mean','?')} / {la.get('total',{}).get('p95','?')} ms")

        rs = report.get("resources", {})
        if rs and "error" not in rs:
            _w("\n[C] RESOURCES")
            _w(f"  QPS            : {rs.get('qps','?')}")
            _w(f"  RAM mean/max   : {rs.get('ram_mb',{}).get('mean','?')} / {rs.get('ram_mb',{}).get('max','?')} MB")
            _w(f"  CPU mean/max   : {rs.get('cpu_pct',{}).get('mean','?')} / {rs.get('cpu_pct',{}).get('max','?')} %")

    log.info(f"Text report saved → {txt_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.5  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step 6 — Evaluation & Benchmarking")
    parser.add_argument("--bench-n",         type=int, default=50,
                        help="Number of queries for the latency benchmark (default: 50)")
    parser.add_argument("--with-tts",        action="store_true",
                        help="Include TTS in latency measurement")
    parser.add_argument("--resource-monitor",action="store_true",
                        help="Run resource monitor for 30s query loop")
    parser.add_argument("--monitor-secs",    type=int, default=30,
                        help="Duration for resource monitoring (default: 30s)")
    parser.add_argument("--intent-only",     action="store_true",
                        help="Only run the intent classification evaluation")
    args = parser.parse_args()

    report = {}

    # ── Always evaluate intent classifier ────────────────────────────────────
    log.info("Evaluating intent classifier …")
    intent_res        = eval_intent()
    report["intent_accuracy"] = intent_res

    if args.intent_only:
        print_report(intent_res, {"n_queries":0,"intent":{},"retrieval":{},"total":{}})
        save_report(report)
        return

    # ── Load components for latency / resource tests ──────────────────────────
    from step2_intent    import load_pipeline
    from step3_retrieval import load_index

    svm_pipe  = load_pipeline()
    kb_index  = load_index()

    # ── Latency ───────────────────────────────────────────────────────────────
    log.info(f"Measuring latency over {args.bench_n} queries …")
    latency_res        = eval_latency(svm_pipe, kb_index,
                                      n=args.bench_n,
                                      with_tts=args.with_tts)
    report["latency"]  = latency_res

    # ── Resources ─────────────────────────────────────────────────────────────
    resource_res = None
    if args.resource_monitor:
        log.info(f"Resource monitoring for {args.monitor_secs}s …")
        resource_res       = eval_resources(svm_pipe, kb_index,
                                            duration_s=args.monitor_secs)
        report["resources"] = resource_res

    print_report(intent_res, latency_res, resource_res)
    save_report(report)


if __name__ == "__main__":
    main()
