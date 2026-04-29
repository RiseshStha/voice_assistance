"""
Model training pipeline with train/test split before SMOTE.
SMOTE is applied only to the training folds and never to the test set.
"""

from __future__ import annotations

import io
import json
import pickle
import sys

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from config import MODEL_PATH, OUTPUTS_DIR, QA_EXCEL

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def load_training_dataframe() -> pd.DataFrame:
    print(f"Loading original cleaned dataset from {QA_EXCEL}...")
    df = pd.read_excel(QA_EXCEL)
    df = df.dropna(subset=["question", "intent"]).copy()
    df["question"] = df["question"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()
    return df


def make_pipeline(estimator):
    return ImbPipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char_wb", sublinear_tf=True)),
            ("svd", TruncatedSVD(random_state=42)),
            ("smote", SMOTE(random_state=42)),
            ("clf", estimator),
        ]
    )


def save_split_summary(y_train: pd.Series, y_test: pd.Series) -> None:
    summary = {
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "train_intent_counts_before_smote": {
            k: int(v) for k, v in y_train.value_counts().sort_index().items()
        },
        "test_intent_counts_untouched": {
            k: int(v) for k, v in y_test.value_counts().sort_index().items()
        },
        "smote_policy": "SMOTE applied only on training folds; test set remains original and untouched.",
    }
    with open(OUTPUTS_DIR / "train_test_split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    df = load_training_dataframe()
    X = df["question"].astype(str)
    y = df["intent"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training rows before SMOTE: {len(X_train)}, untouched test rows: {len(X_test)}")
    save_split_summary(y_train, y_test)

    for stale_name in ("model_rf.pkl", "model_nb.pkl"):
        stale_path = OUTPUTS_DIR / stale_name
        if stale_path.exists():
            stale_path.unlink()
            print(f"Removed stale artifact: {stale_path.name}")

    svm_pipeline = make_pipeline(LinearSVC(max_iter=5000))
    param_grid = {
        "tfidf__ngram_range": [(2, 3), (2, 4)],
        "svd__n_components": [100, 200, 300],
        "clf__C": [0.1, 1, 10],
    }

    print("Running GridSearchCV with SMOTE only on training folds...")
    grid = GridSearchCV(svm_pipeline, param_grid, cv=5, scoring="f1_weighted", n_jobs=1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Best SVM parameters: {grid.best_params_}")

    models = {
        "lr": make_pipeline(LogisticRegression(max_iter=1500, random_state=42)),
        "knn": make_pipeline(KNeighborsClassifier(n_neighbors=7, weights="distance")),
    }

    fixed_svd = grid.best_params_.get("svd__n_components", 300)
    fixed_ngram = grid.best_params_.get("tfidf__ngram_range", (2, 4))
    for pipeline in models.values():
        pipeline.set_params(svd__n_components=fixed_svd, tfidf__ngram_range=fixed_ngram)

    print("Training comparison models with the same train-only SMOTE policy...")
    for name, pipeline in models.items():
        print(f"Training {name.upper()}...")
        pipeline.fit(X_train, y_train)
        path = OUTPUTS_DIR / f"model_{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(pipeline, f)

    print(f"Training complete. Models saved in {OUTPUTS_DIR}")
    print(f"Split summary saved to {OUTPUTS_DIR / 'train_test_split_summary.json'}")


if __name__ == "__main__":
    main()
