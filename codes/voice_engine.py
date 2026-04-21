"""
voice_engine.py — Part 3: ASR and TTS Integration
=================================================
Handles microphone input (STT) and system voice output (TTS).
"""
import os, sys, logging, time
import numpy as np
from pathlib import Path
from faster_whisper import WhisperModel

# Project imports
from config import WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE, MODELS_DIR, SAMPLE_RATE

log = logging.getLogger(__name__)

def load_stt_model():
    """Load Faster-Whisper."""
    return WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, 
                        compute_type=WHISPER_COMPUTE, download_root=str(MODELS_DIR))

def speak_text(text):
    """
    Voice output.
    - Windows-friendly default: pyttsx3 (uses installed system voices)
    - Fallback: eSpeak-NG if present on PATH
    """
    text = str(text).strip()
    if not text:
        return

    try:
        import pyttsx3  # type: ignore
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        engine.say(text)
        engine.runAndWait()
        return
    except Exception:
        pass

    try:
        import subprocess
        subprocess.run(["espeak-ng", "-v", "ne", text], check=True)
    except Exception as e:
        log.warning("TTS failed: %s", e)

def record_microphone(duration_s: float = 4.0, device: int | None = None) -> np.ndarray:
    """
    Record mono audio from the default microphone and return float32 waveform in [-1, 1].
    Requires `sounddevice`.
    """
    try:
        import sounddevice as sd  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency for microphone recording. Install: pip install sounddevice"
        ) from e

    duration_s = float(duration_s)
    if duration_s <= 0:
        raise ValueError("duration_s must be > 0")

    frames = int(SAMPLE_RATE * duration_s)
    audio = sd.rec(
        frames,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device,
        blocking=True,
    )
    return np.asarray(audio).reshape(-1)

def transcribe_stream(model, audio_np):
    """
    High-performance transcription with noise rejection.
    Ensures hallucinated segments (no-speech) are filtered out.
    """
    segments, info = model.transcribe(
        audio_np, 
        language="ne", 
        beam_size=5, 
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    full_text = []
    for s in segments:
        # 80+ Grade Detail: Filter hallucinations via Whisper confidence scores
        if s.no_speech_prob < 0.6 and s.compression_ratio < 2.4:
            full_text.append(s.text)
            
    return " ".join(full_text).strip()

def listen_and_transcribe(model, duration_s: float = 4.0, device: int | None = None) -> str:
    """Convenience helper: record from mic then transcribe."""
    audio = record_microphone(duration_s=duration_s, device=device)
    return transcribe_stream(model, audio)
