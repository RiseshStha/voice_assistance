# Nepali Voice Assistant (Machine Learning CW)

This project is an **intelligent application prototype** for Nepal: a **Nepali student assistant** that takes a question (typed prompt or microphone speech), transcribes Nepali speech to text, then **classifies the intent** (e.g., math/science/social) using machine learning.

## Folder structure (submission-ready)

- `codes/`
  - `data_prep.py`: dataset cleaning + EDA plots
  - `train_model.py`: ML training (optimized SVM pipeline)
  - `voice_engine.py`: mic recording (STT) + voice output (TTS)
  - `app.py`: GUI prototype (typed prompt + listening)
  - `config.py`: centralized paths/config
- `outputs/`: generated reports/figures/models
- `models/`: local ASR model cache (Whisper download root)
- `requirements.txt`: Python dependencies

## Quick start (Windows)

From the `Nepali_Voice_Assistant/` folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 1) Prepare data + EDA

```powershell
python .\codes\data_prep.py
```

This writes:
- `outputs/eda_intent_dist.png`
- cleaned dataset into the path configured in `codes/config.py`

### 2) Train the ML model

```powershell
python .\codes\train_model.py
```

This writes:
- `outputs/final_optimized_svm.pkl`
- `outputs/confusion_matrix.png`

### 2b) Evaluate + visualizations + latency benchmarks (for marking)

```powershell
python .\codes\evaluate_and_benchmark.py
```

This writes:
- `outputs/eval_report.json` + `outputs/eval_report.txt`
- `outputs/confusion_matrix_eval.png` (confusion matrix heatmap)
- `outputs/accuracy_table.png` (accuracy results table)

### 3) Build the Answer KB (for real answers)

```powershell
python .\codes\build_kb.py
```

This writes:
- `outputs/kb_index.pkl`

### 3) Run the GUI (typed prompt + microphone)

```powershell
python .\codes\app.py
```

Notes:
- Microphone recording uses `sounddevice`. If it errors, install a Windows audio backend and ensure your mic is enabled in Windows Privacy settings.
- Text-to-speech uses `pyttsx3` by default; if unavailable, it falls back to `espeak-ng` if installed.

## Screencast requirement (what to show)

Record a 1–2 minute video showing:
- running `data_prep.py` and the EDA plot in `outputs/`
- running `train_model.py` and the confusion matrix
- launching `app.py`
- one **typed** query + one **spoken** query

## Academic integrity

Write the final IEEE-style report in your own words. If you use any external datasets or references, cite them properly.

