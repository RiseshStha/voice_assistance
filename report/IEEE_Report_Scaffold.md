## Abstract

Write 150–250 words: problem context in Nepal, dataset, 3 ML methods, best result, and prototype contribution (GUI + voice).

## 1. Introduction

- Motivation (Nepali student support / accessibility / low-resource language)
- Problem statement: intent classification for Nepali educational questions, plus voice-to-text interface
- Brief related work (2–6 sources): Nepali/Hindi ASR, intent classification, TF-IDF + linear models, SVM for text classification
- Contributions (bullet list): dataset preprocessing, 3-model comparison, best model selection, GUI prototype with typed + voice modes

## 2. Problem and Dataset Description

- Define classes/intents and examples
- Dataset source + size, class distribution, train/test split strategy
- Ethics/limitations (bias, domain coverage)

## 3. Methods

Describe each model briefly (why it fits, assumptions):
- Logistic Regression (baseline linear classifier)
- Linear SVM (strong for high-dimensional sparse text)
- KNN (instance-based baseline)

Optional: PCA/SVD dimensionality reduction and class balancing (e.g., SMOTE) if used.

## 4. Experimental Setup

- Text normalization (Devanagari cleanup)
- Vectorization: TF-IDF (word or char n-grams)
- Hyperparameters (grid search, cross-validation)
- Metrics: accuracy, macro-F1, weighted-F1, confusion matrix
- Implementation environment (Python version, packages, CPU)

## 5. Results

- Table of metrics for the 3 models
- Confusion matrix for the best model
- Short error analysis (2–4 examples of misclassifications)

## 6. Discussion and Conclusions

- Which model worked best and why
- Practical implications for Nepal
- Limitations and future work (bigger dataset, more intents, Nepali TTS voice, streaming VAD)

## References

Use IEEE referencing style. Cite:
- dataset page (Kaggle/UCI/etc.)
- scikit-learn docs / standard papers (SVM, TF-IDF)
- Whisper / faster-whisper paper or repo

