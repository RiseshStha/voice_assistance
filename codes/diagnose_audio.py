"""
diagnose_audio.py  -  Quick audio pipeline diagnostic
======================================================
Run this FIRST to check each stage before launching the full app.
Usage:  python diagnose_audio.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
RECORD_SECONDS = 5

# ── Step 1: List microphone devices ──────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1: Checking microphone devices")
print("="*60)
try:
    import sounddevice as sd
    devices = sd.query_devices()
    print(f"Default input device: {sd.default.device[0]}")
    print("\nAll INPUT devices:")
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            print(f"  [{i}] {d['name']}  ({d['hostapi']})  sr={int(d['default_samplerate'])}")
except Exception as e:
    print(f"  ERROR: {e}")
    print("  Fix: pip install sounddevice")

# ── Step 2: Record 5 seconds of raw audio ────────────────────────────────────
print("\n" + "="*60)
print(f"STEP 2: Recording {RECORD_SECONDS}s of raw audio (speak now!)")
print("="*60)
raw_audio = None
try:
    import sounddevice as sd
    print("  Recording... speak a Nepali question now!")
    raw_audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    raw_audio = raw_audio.flatten()
    rms = float(np.sqrt(np.mean(np.square(raw_audio))))
    peak = float(np.max(np.abs(raw_audio)))
    print(f"  Recorded {len(raw_audio)/SAMPLE_RATE:.1f}s  |  RMS={rms:.5f}  |  Peak={peak:.5f}")
    if rms < 0.001:
        print("  [WARNING] RMS is very low - microphone may not be picking up audio!")
        print("       Check: microphone volume, correct device selected, not muted.")
    elif rms > 0.05:
        print("  [OK] Audio level looks good.")
    else:
        print("  [LOW] Audio level is low but may be OK.")
except Exception as e:
    print(f"  ERROR recording: {e}")

# ── Step 3: Save raw WAV for manual inspection ────────────────────────────────
print("\n" + "="*60)
print("STEP 3: Saving raw WAV (check this file in Audacity!)")
print("="*60)
if raw_audio is not None:
    import wave
    raw_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "diagnose_raw.wav")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    pcm = np.clip(raw_audio, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    with wave.open(raw_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    print(f"  Saved: {os.path.abspath(raw_path)}")
    print("  -> Open this file in any audio player to hear what was captured.")

# -- Step 4: Test noisereduce --------------------------------------------------
print("\n" + "="*60)
print("STEP 4: Testing noisereduce")
print("="*60)
if raw_audio is not None:
    try:
        import noisereduce as nr
        denoised = nr.reduce_noise(y=raw_audio, sr=SAMPLE_RATE, stationary=True, prop_decrease=0.55)
        rms_before = float(np.sqrt(np.mean(np.square(raw_audio))))
        rms_after = float(np.sqrt(np.mean(np.square(denoised))))
        ratio = rms_after / rms_before if rms_before > 1e-8 else 0
        print(f"  RMS before: {rms_before:.5f}  |  RMS after: {rms_after:.5f}  |  Ratio: {ratio:.2f}")
        if ratio < 0.1:
            print("  [WARNING] noisereduce removed >90% of signal - speech is being destroyed!")
        else:
            print("  [OK] noisereduce preserved signal reasonably.")
    except ImportError:
        print("  noisereduce not installed (OK, will be skipped in pipeline)")
    except Exception as e:
        print(f"  noisereduce error: {e}")

# ── Step 5: Test Faster-Whisper transcription ─────────────────────────────────
print("\n" + "="*60)
print("STEP 5: Testing Faster-Whisper transcription")
print("="*60)
if raw_audio is not None and raw_audio.size > 0:
    try:
        from faster_whisper import WhisperModel
        from config import MODELS_DIR

        print("  Loading Whisper 'base' model (may take a moment)...")
        model = WhisperModel("base", device="cpu", compute_type="int8",
                             download_root=str(MODELS_DIR), num_workers=1)

        print("  Transcribing raw audio (no preprocessing)...")
        segments, info = model.transcribe(
            raw_audio,
            language="ne",
            task="transcribe",
            beam_size=5,
            temperature=0.0,
            vad_filter=True,
        )
        texts = []
        for seg in segments:
            avg_lp = getattr(seg, 'avg_log_prob', getattr(seg, 'avg_logprob', -99))
            no_sp  = getattr(seg, 'no_speech_prob', 0.0)
            print(f"    Segment: no_speech={no_sp:.2f}, "
                  f"log_prob={avg_lp:.2f}, text='{seg.text.strip()}'")
            texts.append(seg.text.strip())
        result = " ".join(texts).strip()
        print(f"\n  [OK] Transcription result: '{result}'")
        if not result:
            print("  [WARNING] Empty result - Whisper VAD removed all audio.")
            print("  --- Retrying with vad_filter=False (full audio) ---")
            segments2, _ = model.transcribe(
                raw_audio,
                language="ne",
                task="transcribe",
                beam_size=5,
                temperature=0.0,
                vad_filter=False,
                log_prob_threshold=-3.5,
                no_speech_threshold=0.65,
                condition_on_previous_text=False,
            )
            texts2 = []
            for seg in segments2:
                avg_lp = getattr(seg, 'avg_log_prob', getattr(seg, 'avg_logprob', -99))
                no_sp  = getattr(seg, 'no_speech_prob', 0.0)
                print(f"    Segment: no_speech={no_sp:.2f}, "
                      f"log_prob={avg_lp:.2f}, text='{seg.text.strip()}'")
                texts2.append(seg.text.strip())
            result2 = " ".join(texts2).strip()
            print(f"  No-VAD result: '{result2}'")
            if result2:
                print("  [CONFIRMED] vad_filter=False fixes the issue!")
                print("  The new Strategy 0 in voice_engine.py will handle this.")
            else:
                print("  [INFO] Still empty - check Windows mic volume (Settings > Sound).")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n" + "="*60)
print("DIAGNOSIS COMPLETE")
print("="*60)
print("Files to inspect:")
print("  - outputs/diagnose_raw.wav      (raw mic capture - before preprocessing)")
print("  - outputs/last_voice_input.wav  (after preprocessing - generated by app)")
print("")
print("If diagnose_raw.wav sounds fine but last_voice_input.wav sounds bad,")
print("the problem is in _preprocess_audio(). If diagnose_raw.wav is silent,")
print("the problem is the microphone setup (wrong device or too quiet).")
