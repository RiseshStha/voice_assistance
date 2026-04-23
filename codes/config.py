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

# ── Project roots ────────────────────────────────────────────────────────────
# This repo has historically had datasets either:
# - inside Nepali_Voice_Assistant/data/..., OR
# - at the workspace root data/...
# We support BOTH to avoid "file not found" when running scripts.
PROJECT_ROOT = Path(__file__).resolve().parent.parent           # .../Nepali_Voice_Assistant
WORKSPACE_ROOT = PROJECT_ROOT.parent                            # .../Research_Assignment
ROOT = PROJECT_ROOT  # Backward compatible name used across scripts

def _pick_existing_dir(*candidates: Path) -> Path:
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]

# ── Data ─────────────────────────────────────────────────────────────────────
DATA_ROOT = _pick_existing_dir(PROJECT_ROOT / "data", WORKSPACE_ROOT / "data")
DATA_DIR        = DATA_ROOT / "data_for_training"
AUDIO_DIR       = DATA_DIR / "audio_data"
QA_EXCEL        = DATA_DIR / "question_answer_clean.xlsx"
CLEAN_XL        = QA_EXCEL  # Alias for training scripts

# Raw dataset (we prefer the newest "update" file if present)
RAW_QA_EXCEL_CANDIDATES = [
    DATA_DIR / "question_answer_final_update.xlsx",
    DATA_DIR / "question_answer_final_update.xls",
    DATA_DIR / "question_answer_final_updated.xlsx",
    DATA_DIR / "question_answer_final_updated.xls",
    DATA_DIR / "question_answer_final.xlsx",
    DATA_DIR / "question_answer_final.xls",
]

def pick_raw_qa_excel() -> Path:
    for p in RAW_QA_EXCEL_CANDIDATES:
        if p.exists():
            return p
    # Default to the first candidate; downstream will raise a clear error
    return RAW_QA_EXCEL_CANDIDATES[0]

# ── Outputs ───────────────────────────────────────────────────────────────────
OUTPUTS_DIR     = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Models ────────────────────────────────────────────────────────────────────
MODELS_DIR      = PROJECT_ROOT / "models"
WHISPER_MODEL_SIZE = "base"  # ~140MB - much better for Nepali than 'tiny'
WHISPER_DEVICE     = "cpu"
WHISPER_COMPUTE    = "int8"  # highly compressed integer quantization for Raspberry Pi CPU

# Unified names for optimized scripts
MODEL_PATH      = OUTPUTS_DIR / "final_optimized_svm.pkl"
KB_INDEX_PATH   = OUTPUTS_DIR / "kb_index.pkl"

# Legacy paths (for backward compatibility during migration)
SVM_MODEL_PATH  = MODEL_PATH
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
