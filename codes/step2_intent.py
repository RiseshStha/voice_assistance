"""
STEP 2 — Intent Classification Module
=======================================
Vectorises Nepali text with TF-IDF and trains a LinearSVC (SVM) classifier
on the 2,098 labelled Q&A samples.

What it produces
----------------
  outputs/tfidf_vectorizer.pkl   — fitted TF-IDF vectoriser
  outputs/intent_svm.pkl         — trained LinearSVC model
  outputs/intent_report.json     — classification report + confusion matrix

Usage
-----
  # Train (80/20 split)
  python step2_intent.py --train

  # Predict intent for a single text string
  python step2_intent.py --predict "भूकम्प भनेको के हो?"

  # Evaluate on a custom test CSV
  python step2_intent.py --eval-csv path/to/test.csv
"""

import sys, json, pickle, logging, argparse
from pathlib import Path

import numpy  as np
import pandas as pd
from sklearn.pipeline           import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm                import LinearSVC
from sklearn.model_selection    import train_test_split, cross_val_score
from sklearn.metrics            import (classification_report,
                                        confusion_matrix,
                                        accuracy_score)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import QA_EXCEL, SVM_MODEL_PATH, TFIDF_PATH, OUTPUTS_DIR, INTENTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 2.1  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(excel_path: Path = QA_EXCEL) -> pd.DataFrame:
    """Load the Q&A Excel and return a clean DataFrame."""
    log.info(f"Loading dataset from {excel_path} …")
    df = pd.read_excel(excel_path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"question", "intent"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Excel is missing columns: {missing}")

    df = df.dropna(subset=["question", "intent"])
    df["question"] = df["question"].str.strip()
    df["intent"]   = df["intent"].str.strip()

    log.info(f"Loaded {len(df)} samples | {df['intent'].nunique()} unique intents")
    log.info("Intent distribution:\n" +
             df["intent"].value_counts().to_string())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.2  Build & train pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    TF-IDF (char + word n-grams) → LinearSVC pipeline.

    Notes
    -----
    * analyzer='char_wb' captures Devanagari morphology well even without a
      Nepali tokeniser.  We combine char-ngrams with word-unigrams.
    * C=1.0 is a safe default; tune via --cross-val if needed.
    """
    tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),        # character 2-to-4-grams
        min_df=1,
        max_features=50_000,
        sublinear_tf=True,
    )
    svm = LinearSVC(
        C=1.0,
        max_iter=2000,
        class_weight="balanced",   # handles class imbalance gracefully
    )
    return Pipeline([("tfidf", tfidf), ("svm", svm)])


# ─────────────────────────────────────────────────────────────────────────────
# 2.3  Train
# ─────────────────────────────────────────────────────────────────────────────

def train(test_size: float = 0.20, random_state: int = 42,
          cross_val: bool = False) -> dict:
    """
    Train the classifier and save model artefacts.

    Returns
    -------
    dict with accuracy, report, confusion_matrix
    """
    df   = load_dataset()
    X    = df["question"].tolist()
    y    = df["intent"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    log.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    pipe = build_pipeline()
    log.info("Training pipeline …")
    pipe.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred   = pipe.predict(X_test)
    acc      = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred, output_dict=True,
                                     zero_division=0)
    cm       = confusion_matrix(y_test, y_pred,
                                labels=pipe.classes_).tolist()
    labels   = list(pipe.classes_)

    log.info(f"\n{classification_report(y_test, y_pred, zero_division=0)}")
    log.info(f"Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # ── Optional cross-val ────────────────────────────────────────────────────
    cv_mean = None
    if cross_val:
        log.info("Running 5-fold cross-validation (may take ~30s) …")
        scores  = cross_val_score(pipe, X, y, cv=5, scoring="accuracy", n_jobs=-1)
        cv_mean = float(scores.mean())
        log.info(f"CV accuracy: {scores} → mean={cv_mean:.4f}")

    # ── Save artefacts ────────────────────────────────────────────────────────
    _save_pipeline(pipe)

    results = {
        "test_accuracy":      round(acc, 6),
        "train_samples":      len(X_train),
        "test_samples":       len(X_test),
        "report":             report,
        "confusion_matrix":   cm,
        "labels":             labels,
        "cv_accuracy_mean":   cv_mean,
    }

    out = OUTPUTS_DIR / "intent_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log.info(f"Report saved → {out}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2.4  Save / load model
# ─────────────────────────────────────────────────────────────────────────────

def _save_pipeline(pipe: Pipeline):
    """Save the full sklearn Pipeline as a single pickle."""
    SVM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Save as one unified file for convenience
    with open(SVM_MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)
    log.info(f"Model saved → {SVM_MODEL_PATH}")


def load_pipeline(model_path: Path = SVM_MODEL_PATH) -> Pipeline:
    """Load the trained pipeline from disk."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}.\n"
            "Run:  python step2_intent.py --train"
        )
    with open(model_path, "rb") as f:
        pipe = pickle.load(f)
    log.info(f"Loaded intent model from {model_path}")
    return pipe


# ─────────────────────────────────────────────────────────────────────────────
# 2.5  Predict single text
# ─────────────────────────────────────────────────────────────────────────────

def predict_intent(text: str, pipe: Pipeline = None) -> dict:
    """
    Predict the intent of a single Nepali text string.

    Returns
    -------
    { "intent": str, "confidence": float, "all_scores": dict }
    """
    if pipe is None:
        pipe = load_pipeline()

    text      = text.strip()
    intent    = pipe.predict([text])[0]

    # LinearSVC doesn't emit probabilities; use decision-function scores instead
    scores_raw = pipe.decision_function([text])[0]
    labels     = list(pipe.classes_)
    scores     = {lbl: round(float(s), 4)
                  for lbl, s in zip(labels, scores_raw)}

    # Normalise to [0,1] softmax-style for a confidence proxy
    exp_s  = np.exp(scores_raw - scores_raw.max())
    softmax = exp_s / exp_s.sum()
    conf   = float(softmax[labels.index(intent)])

    return {
        "text":       text,
        "intent":     intent,
        "confidence": round(conf, 4),
        "all_scores": scores,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2.6  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step 2 — Nepali Intent Classifier")
    parser.add_argument("--train",     action="store_true",
                        help="Train the SVM classifier and save model")
    parser.add_argument("--predict",   type=str, default=None,
                        help="Predict intent for a given Nepali text string")
    parser.add_argument("--eval-csv",  type=str, default=None,
                        help="Evaluate on a CSV file with 'question' & 'intent' columns")
    parser.add_argument("--cross-val", action="store_true",
                        help="Run 5-fold cross-validation during training")
    parser.add_argument("--test-size", type=float, default=0.20,
                        help="Fraction of data for testing (default: 0.20)")
    args = parser.parse_args()

    if args.train:
        train(test_size=args.test_size, cross_val=args.cross_val)
        return

    if args.predict:
        pipe   = load_pipeline()
        result = predict_intent(args.predict, pipe)
        print(f"\n  Text    : {result['text']}")
        print(f"  Intent  : {result['intent']}")
        print(f"  Conf    : {result['confidence']:.2%}")
        print(f"  Scores  : " +
              "  ".join(f"{k}={v:.3f}" for k, v in sorted(result['all_scores'].items(),
                                                            key=lambda x: -x[1])))
        return

    if args.eval_csv:
        df = pd.read_csv(args.eval_csv, dtype=str)
        pipe   = load_pipeline()
        y_pred = pipe.predict(df["question"].tolist())
        y_true = df["intent"].tolist()
        print(classification_report(y_true, y_pred, zero_division=0))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
