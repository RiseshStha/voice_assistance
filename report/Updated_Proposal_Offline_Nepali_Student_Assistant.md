# An Offline Voice-Based Nepali Student Assistant Using Classical Machine Learning (SVM) for Low-Resource Immediate Systems  
**Project Proposal (Updated to match implementation)**

## Abstract
This project proposes and develops a fully offline, voice-based Nepali student assistant for low-resource environments in Nepal where internet connectivity and digital learning tools are limited. The prototype integrates offline automatic speech recognition (ASR), classical machine learning–based intent classification, and local knowledge-base retrieval to provide immediate answers to students’ subject questions. Unlike cloud-based assistants, the system runs on CPU-only hardware and avoids network dependence. The core ML component evaluates and compares three classification techniques—Logistic Regression, K-Nearest Neighbors (KNN), and Linear Support Vector Machine (LinearSVC)—using TF-IDF character n-gram features. The final system is implemented as a Python application with a desktop GUI, offline speech-to-text, and offline text-to-speech, targeting practical deployment on modest computers and edge devices.

## 1. Introduction
Rural and low-resource schools in Nepal frequently face barriers such as limited internet access, fewer qualified subject teachers, and lack of Nepali-language educational technology. Most commercial voice assistants depend on cloud services and do not reliably support Nepali for offline learning. This creates a clear opportunity for a lightweight Nepali voice assistant that works without internet and can run on CPU-only devices.

Recent progress in efficient ASR implementations (e.g., Faster-Whisper with CPU INT8 inference) enables offline Nepali transcription at practical speeds. For the NLP component, classical ML (TF-IDF + linear classifiers) remains attractive for low-resource settings because it provides strong baselines with small model size and fast inference.

## 2. Problem Statement
Students in low-resource Nepali contexts face:
- Limited or unreliable internet access, preventing use of cloud assistants
- Difficulty getting immediate subject help outside class time
- Lack of Nepali-first educational tools capable of running offline

The project addresses the problem: **How can we build an offline Nepali student assistant that understands student questions (by intent) and retrieves appropriate answers from a local knowledge base, with real-time responsiveness on CPU-only hardware?**

## 3. Objectives
### 3.1 Main Objective
To design, develop, and evaluate a fully offline Nepali student assistant that supports both typed and voice queries by combining offline ASR, intent classification using classical ML, and local knowledge-base retrieval.

### 3.2 Specific Objectives
1. Build/curate a Nepali educational Q&A dataset labeled by subject intent.
2. Implement consistent preprocessing and TF-IDF character n-gram feature extraction.
3. Train and compare **three** ML techniques (LogReg, KNN, Linear SVM) for intent classification.
4. Select an optimized classifier (SVM pipeline) using cross-validation and imbalance handling.
5. Implement offline retrieval from a locally stored knowledge base conditioned on predicted intent.
6. Integrate offline ASR (Faster-Whisper) and offline Nepali TTS (eSpeak-NG) into a Python GUI prototype.
7. Benchmark accuracy/F1 and runtime latency to validate real-time offline feasibility.

## 4. Research Questions
1. Can a fully offline Nepali student assistant provide real-time responses on CPU-only hardware without internet connectivity?
2. Which classical ML technique (among LogReg, KNN, Linear SVM) performs best for Nepali educational intent classification under the same TF-IDF feature setup?
3. How does class imbalance influence intent classification performance and what mitigation (e.g., SMOTE, class weighting) is most effective?

## 5. Literature Review (Short)
- **Classical ML for text classification**: Linear SVM and Logistic Regression are strong baselines for sparse TF-IDF text representations, especially in constrained environments.
- **Class imbalance**: SMOTE is a standard oversampling technique to improve minority-class recall when data collection is limited.
- **Offline ASR**: Whisper-style models provide multilingual ASR; Faster-Whisper improves CPU inference efficiency, enabling offline usage.
- **Offline retrieval**: TF-IDF cosine similarity / BM25 remain effective for domain-limited knowledge bases and require minimal compute.

## 6. Methodology
### 6.1 System Architecture (Offline Pipeline)
1. **Speech Input** (microphone) or typed input  
2. **ASR (Speech → Text)**: Faster-Whisper (Nepali language) on CPU INT8  
3. **Text preprocessing**: Devanagari cleanup, noise removal  
4. **Intent classification**: TF-IDF char n-grams + (LogReg / KNN / Linear SVM)  
5. **Answer retrieval**: intent-conditioned TF-IDF cosine similarity search in local KB  
6. **TTS**: eSpeak-NG Nepali voice (fallback supported)  
7. **GUI**: desktop app for interaction

### 6.2 Dataset Plan
- Source: Nepali primary school textbook questions/answers and curated educational content.
- Labeling: subject-intent labels (e.g., science, social, nepali language, health, physical education, arts).
- Target: a few thousand samples; ensure representation across intents; document imbalance.

### 6.3 Experimental Design
- Train/test split: stratified 80/20
- Feature extraction: TF-IDF (`char_wb`, n-grams)
- Models: Logistic Regression, KNN, Linear SVM
- Optimization (SVM): optional TruncatedSVD + SMOTE + GridSearchCV
- Metrics: accuracy, macro-F1, weighted-F1; confusion matrix; latency benchmark (ms/query)

## 7. Ethical, Social, and Professional Considerations
- **Privacy**: offline-by-design avoids transmitting student audio/text to cloud services.
- **Safety**: retrieval-based answers from a curated KB reduce hallucination risks compared to generative systems.
- **Bias**: dataset coverage may reflect textbook bias; class imbalance may disadvantage minority intents—mitigate via class weighting/SMOTE and report per-class metrics.
- **Copyright**: if textbooks are used, the dataset should be handled for academic use only; do not redistribute copyrighted content without permission.

## 8. Expected Outcomes
1. A functional offline Nepali student assistant prototype (GUI + voice + retrieval).
2. Comparative evaluation of three ML techniques on the same dataset/features.
3. Reproducible experiment artifacts (metrics, confusion matrix, latency report).
4. A finalized report written in IEEE-style structure with appendices showing evidence of experiments.

## References (starter list)
- N. V. Chawla et al., “SMOTE: Synthetic Minority Over-sampling Technique,” *JAIR*, 2002.  
- A. Radford et al., “Robust Speech Recognition via Large-Scale Weak Supervision,” *ICML*, 2023.  
- T. Joachims, “Text Categorization with Support Vector Machines,” *ECML*, 1998.  
- F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” *JMLR*, 2011.

