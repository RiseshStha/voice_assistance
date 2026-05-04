"""
Nepali Voice Engine (Offline) - Faster-Whisper Small Version
Optimized for low-resource devices (Raspberry Pi friendly)
"""

import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import threading
import subprocess

SAMPLE_RATE = 16000


# ---------------------------
# LOAD MODEL
# ---------------------------
def load_stt_model():
    """
    Load optimized Whisper small model for CPU usage.
    """
    return WhisperModel(
        "small",              # balanced accuracy + speed
        device="cpu",
        compute_type="int8"   # important for Raspberry Pi
    )


# ---------------------------
# RECORD AUDIO
# ---------------------------
def record_audio(duration=5, device=None):
    print("Listening...")

    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device
    )

    sd.wait()
    return audio.flatten()


# ---------------------------
# TEXT CLEANING (VERY IMPORTANT)
# ---------------------------
def clean_text(text: str) -> str:
    text = text.strip().lower()

    # common Nepali speech errors from Whisper
    replacements = {
        "बिज्ञान": "विज्ञान",
        "विज्यान": "विज्ञान",
        "सिकने": "सिक्ने",
        "सिखने": "सिक्ने",
        "पहिलो": "पहिलो",
        "पहलो": "पहिलो",
        "चरण": "चरण",
        "के हो": "के हो",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    # remove extra spaces
    text = " ".join(text.split())

    return text



# TRANSCRIPTION

def transcribe(model, audio):
    print("Transcribing...")

    segments, _ = model.transcribe(
        audio,
        language="ne",         # Nepali
        beam_size=5,           # improves accuracy
        best_of=5,
        temperature=0.0,
        vad_filter=True,       # removes silence noise
        initial_prompt=(
            "यो नेपाली शैक्षिक प्रश्न हो। "
            "उदाहरण: विज्ञान सिक्ने पहिलो चरण के हो?"
        )
    )

    text = " ".join([seg.text for seg in segments]).strip()
    return clean_text(text)



# MAIN PIPELINE

def listen_and_transcribe(model, duration_s=5, device=None, stop_event=None):
    try:
        audio = record_audio(duration_s, device)

        if len(audio) == 0:
            return ""

        text = transcribe(model, audio)

        if not text:
            return ""

        return text

    except Exception as e:
        print("STT Error:", e)
        return ""


# TTS (eSpeak fallback)

def speak_text(text: str):
    try:
        subprocess.run(["espeak", "-v", "en", text])
    except Exception:
        print("TTS failed")


def speak_text_async(text: str):
    threading.Thread(target=speak_text, args=(text,), daemon=True).start()



# AUDIO DEVICES

def list_input_devices():
    devices = sd.query_devices()
    return [
        {
            "id": i,
            "name": dev["name"]
        }
        for i, dev in enumerate(devices)
        if dev["max_input_channels"] > 0
    ]


def choose_default_input_device():
    return sd.default.device[0]



# OPTIONAL COMPATIBILITY HOOKS (for your app.py)

def has_native_nepali_tts():
    return False


def calibrate_noise_profile(duration_s=2.0):
    print("🔇 Calibrating noise... (placeholder)")