"""
train_model.py — Part 2: Machine Learning Pipeline
==================================================
Technical contribution: Implements GridSearch for SVM optimization 
and SMOTE for high-performance class balancing.
"""
import pandas as pd
import numpy as np
import pickle, sys, io
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Scikit-learn & Imblearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Project Settings
from config import MODEL_PATH, CLEAN_XL, ROOT

def main():
    print(f"Loading cleaned dataset from {CLEAN_XL}...")
    df = pd.read_excel(CLEAN_XL)
    X, y = df['question'].astype(str), df['intent']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Executing Hyperparameter Optimization (Grid Search)...")
    
    pipeline = ImbPipeline([
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

    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    # Save Model
    best_model = grid.best_estimator_
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    
    # Evaluation Visualize
    print("\nTraining Complete. Best Parameters:", grid.best_params_)
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', xticklabels=grid.classes_, yticklabels=grid.classes_)
    plt.title("Model Confusion Matrix")
    plt.savefig(ROOT / "outputs" / "confusion_matrix.png")
    
    # BONUS: Interpretability (Discussion Section)
    weights = best_model.named_steps['svc'].coef_ @ best_model.named_steps['svd'].components_
    feature_names = best_model.named_steps['tfidf'].get_feature_names_out()
    
    print("\n--- Key Interpretability Metadata ---")
    for idx, name in enumerate(grid.classes_):
        top = np.argsort(weights[idx])[-5:]
        print(f"{name}: {feature_names[top]}")

if __name__ == "__main__":
    main()
