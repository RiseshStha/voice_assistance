"""
train_model.py — Part 2: Machine Learning Pipeline
==================================================
Technical contribution: Implements GridSearch for SVM optimization
and SMOTE for high-performance class balancing. Also trains baseline
models for comparative study (LR, KNN).
"""
import pandas as pd
import pickle, sys, io

# Scikit-learn & Imblearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Project Settings
from config import MODEL_PATH, CLEAN_XL, OUTPUTS_DIR

def main():
    print(f"Loading cleaned dataset from {CLEAN_XL}...")
    df = pd.read_excel(CLEAN_XL)
    X, y = df['question'].astype(str), df['intent']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Executing Hyperparameter Optimization (Grid Search) for SVM...")

    # Remove stale comparison model artifacts from older experiment setups.
    for stale_name in ("model_rf.pkl", "model_nb.pkl"):
        stale_path = OUTPUTS_DIR / stale_name
        if stale_path.exists():
            stale_path.unlink()
            print(f"Removed stale artifact: {stale_path.name}")
    
    svm_pipeline = ImbPipeline([
        ('tfidf', TfidfVectorizer(analyzer="char_wb", sublinear_tf=True)),
        ('svd', TruncatedSVD(random_state=42)),
        ('smote', SMOTE(random_state=42)),
        ('svc', LinearSVC(max_iter=5000, class_weight='balanced'))
    ])

    param_grid = {
        'tfidf__ngram_range': [(2, 3), (2, 4)],
        'svd__n_components': [300],
        'svc__C': [0.1, 1, 10]
    }

    # Use a single process for compatibility with restricted Windows environments.
    grid = GridSearchCV(svm_pipeline, param_grid, cv=5, scoring='f1_weighted', n_jobs=1)
    grid.fit(X_train, y_train)
    
    # Save Model
    best_model = grid.best_estimator_
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    print("\nSVM Training Complete. Best Parameters:", grid.best_params_)

    # Train Comparative Models
    print("\nTraining Comparative Models...")
    
    models = {
        'lr': ImbPipeline([
            ('tfidf', TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4), sublinear_tf=True)),
            ('svd', TruncatedSVD(n_components=300, random_state=42)),
            ('smote', SMOTE(random_state=42)),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
        ]),
        'knn': ImbPipeline([
            ('tfidf', TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4), sublinear_tf=True)),
            ('svd', TruncatedSVD(n_components=300, random_state=42)),
            ('smote', SMOTE(random_state=42)),
            ('clf', KNeighborsClassifier(n_neighbors=7, weights='distance'))
        ])
    }

    for name, pipeline in models.items():
        print(f"Training {name.upper()}...")
        pipeline.fit(X_train, y_train)
        path = OUTPUTS_DIR / f"model_{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(pipeline, f)
    
    print("\nAll Models Trained and Saved to /outputs.")

if __name__ == "__main__":
    main()
