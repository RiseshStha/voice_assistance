"""
Shared configuration for the Nepali Student Voice Assistant pipeline.
All step scripts import from here so paths are defined in ONE place.
"""

import sys, io

# ── Windows UTF-8 fix: prevent UnicodeEncodeError for Devanagari / arrow chars ──
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from pathlib import Path

# ── Project root (parent of codes/) ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── Data ─────────────────────────────────────────────────────────────────────
DATA_DIR        = ROOT / "data" / "data_for_training"
AUDIO_DIR       = DATA_DIR / "audio_data"
QA_EXCEL        = DATA_DIR / "question_answer_final.xlsx"

# ── Outputs ───────────────────────────────────────────────────────────────────
OUTPUTS_DIR     = ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Models ────────────────────────────────────────────────────────────────────
MODELS_DIR      = ROOT / "models"
VOSK_MODEL_DIR  = MODELS_DIR / "vosk-model-small-hi-0.22"
VOSK_MODEL_URL  = "https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip"

SVM_MODEL_PATH  = OUTPUTS_DIR / "intent_svm.pkl"
TFIDF_PATH      = OUTPUTS_DIR / "tfidf_vectorizer.pkl"

WAV_CACHE_DIR   = OUTPUTS_DIR / "wav_cache"   # converted .m4a → .wav stored here
WAV_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Audio ────────────────────────────────────────────────────────────────────
SAMPLE_RATE     = 16_000   # Vosk Nepali model requires 16 kHz mono PCM-16

# ── Intent labels (7 classes) ─────────────────────────────────────────────────
INTENTS = [
    "math_question",
    "science_question",
    "social_question",
    "nepali_language_question",
    "health_question",
    "physical_education_question",
    "arts_question",
]
