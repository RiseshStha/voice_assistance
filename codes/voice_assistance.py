"""
STEP 1 — Speech-to-Text (ASR) Module
=====================================
Offline Nepali ASR using Vosk (vosk-model-small-ne-0.15, ~40 MB).

Features
--------
* Converts .m4a / .mp3 → 16 kHz mono WAV automatically (cached in outputs/wav_cache/)
* Batch-transcribes every audio file in data/data_for_training/audio_data/
* Computes Word Error Rate (WER) when a references JSON is provided
* Real-time microphone mode (--mic flag)
* All results saved to outputs/asr_results.json

Usage
-----
  # 1. Download model ONCE (needs internet)
  python voice_assistance.py --download

  # 2. Batch transcribe your audio files
  python voice_assistance.py

  # 3. With WER references
  python voice_assistance.py --references outputs/references.json

  # 4. Live microphone
  python voice_assistance.py --mic
"""

import os, sys, json, wave, time, logging, argparse
from pathlib import Path
from typing import Optional

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    AUDIO_DIR, VOSK_MODEL_DIR, VOSK_MODEL_URL,
    OUTPUTS_DIR, WAV_CACHE_DIR, SAMPLE_RATE,
)

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.1  Model bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def download_model(model_dir: Path = VOSK_MODEL_DIR) -> bool:
    """Download and unzip the Vosk Nepali small model (run once, then go offline)."""
    if model_dir.exists() and any(model_dir.iterdir()):
        log.info(f"Model already present at: {model_dir}")
        return True

    log.info("Model not found — downloading (~40 MB)…")
    try:
        import urllib.request, zipfile
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        zip_path = model_dir.parent / "vosk_ne_model.zip"

        urllib.request.urlretrieve(VOSK_MODEL_URL, zip_path)
        log.info("Extracting…")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(model_dir.parent)
        zip_path.unlink()
        log.info(f"Model ready → {model_dir}")
        return True
    except Exception as exc:
        log.error(f"Download failed: {exc}")
        log.error(f"  1. Manually download: {VOSK_MODEL_URL}")
        log.error(f"  2. Unzip to:          {model_dir}")
        return False


def load_model(model_dir: Path = VOSK_MODEL_DIR):
    """Load Vosk model. Returns a vosk.Model instance."""
    try:
        from vosk import Model
    except ImportError:
        raise RuntimeError("Vosk not installed.  Run: pip install vosk")

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Vosk model not found at {model_dir}\n"
            "Run:  python voice_assistance.py --download"
        )
    log.info(f"Loading Vosk model from {model_dir} …")
    t0 = time.time()
    model = Model(str(model_dir))
    log.info(f"Model loaded in {time.time()-t0:.2f}s")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 1.2  Audio helpers  (m4a / mp3 → 16 kHz mono WAV)
# ─────────────────────────────────────────────────────────────────────────────

def to_wav16k(src: Path, cache_dir: Path = WAV_CACHE_DIR) -> Path:
    """
    Convert any audio file (wav, m4a, mp3 …) to 16 kHz mono 16-bit WAV.
    Converted files are cached — re-conversion is skipped on subsequent runs.
    Requires: librosa + soundfile  (pip install librosa soundfile)
    """
    dst = cache_dir / (src.stem + "_16k.wav")
    if dst.exists():
        return dst          # already converted

    try:
        import librosa, soundfile as sf
        audio, _ = librosa.load(str(src), sr=SAMPLE_RATE, mono=True)
        sf.write(str(dst), audio, SAMPLE_RATE, subtype="PCM_16")
        log.debug(f"Converted {src.name} → {dst.name}")
        return dst
    except Exception as exc:
        raise RuntimeError(f"Audio conversion failed for {src}: {exc}\n"
                           "Make sure ffmpeg is on PATH for m4a support.")


def wav_info(path: Path) -> dict:
    """Return basic WAV metadata."""
    with wave.open(str(path), "rb") as wf:
        n = wf.getnframes()
        r = wf.getframerate()
        return {
            "channels": wf.getnchannels(),
            "rate":     r,
            "frames":   n,
            "duration": n / r,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 1.3  Core transcription
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_wav(wav_path: Path, model) -> dict:
    """
    Transcribe a single WAV file with Vosk.

    Returns dict with keys:
        file, transcript, confidence, latency_s, words
    """
    from vosk import KaldiRecognizer

    # Ensure correct format
    info = wav_info(wav_path)
    if info["rate"] != SAMPLE_RATE or info["channels"] != 1:
        wav_path = to_wav16k(wav_path)

    rec = KaldiRecognizer(model, SAMPLE_RATE)
    rec.SetWords(True)

    t0         = time.time()
    full_text  = []
    all_words  = []

    with wave.open(str(wav_path), "rb") as wf:
        while True:
            data = wf.readframes(4000)   # ~0.25 s chunks → low RAM
            if not data:
                break
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                if res.get("text"):
                    full_text.append(res["text"])
                if "result" in res:
                    all_words.extend(res["result"])

    final = json.loads(rec.FinalResult())
    if final.get("text"):
        full_text.append(final["text"])
    if "result" in final:
        all_words.extend(final["result"])

    latency    = time.time() - t0
    transcript = " ".join(full_text).strip()
    confs      = [w.get("conf", 0) for w in all_words]
    avg_conf   = sum(confs) / len(confs) if confs else 0.0

    return {
        "file":       wav_path.name,
        "transcript": transcript,
        "confidence": round(avg_conf, 4),
        "latency_s":  round(latency, 3),
        "words":      all_words,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1.4  WER (pure Python — no network needed)
# ─────────────────────────────────────────────────────────────────────────────

def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Word Error Rate via dynamic programming.
    WER = (S + D + I) / N
    """
    ref = reference.strip().split()
    hyp = hypothesis.strip().split()
    if not ref:
        return 0.0 if not hyp else 1.0

    n, m = len(ref), len(hyp)
    dp   = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j]   + 1,
                dp[i][j-1]   + 1,
                dp[i-1][j-1] + cost,
            )
    return round(dp[n][m] / n, 4)


# ─────────────────────────────────────────────────────────────────────────────
# 1.5  Batch evaluation
# ─────────────────────────────────────────────────────────────────────────────

def batch_transcribe(audio_dir: Path, model, references: dict = None) -> list:
    """
    Transcribe all audio files (.wav, .m4a, .mp3) in audio_dir.

    Parameters
    ----------
    references : dict mapping  filename → reference_transcript  (optional, for WER)

    Returns
    -------
    list of result dicts
    """
    # Accept wav + m4a + mp3
    files = sorted(
        f for f in audio_dir.iterdir()
        if f.suffix.lower() in {".wav", ".m4a", ".mp3"}
    )

    if not files:
        log.warning(f"No audio files found in {audio_dir}")
        return []

    log.info(f"Found {len(files)} audio file(s) in {audio_dir}")
    results = []

    for src in files:
        log.info(f"  Processing: {src.name}")
        try:
            # Convert to WAV if needed
            if src.suffix.lower() != ".wav":
                wav = to_wav16k(src)
            else:
                wav = src

            res = transcribe_wav(wav, model)
            res["source_file"] = src.name   # keep original filename

            if references and src.name in references:
                ref        = references[src.name]
                res["reference"] = ref
                res["wer"]       = compute_wer(ref, res["transcript"])

            results.append(res)

            status = (f"    ✓ {src.name:35s} | "
                      f"conf={res['confidence']:.2f} | "
                      f"{res['latency_s']:.2f}s")
            if "wer" in res:
                status += f" | WER={res['wer']:.2%}"
            log.info(status)

        except Exception as exc:
            log.error(f"  ✗ {src.name}: {exc}")
            results.append({"file": src.name, "error": str(exc)})

    return results


def save_results(results: list, out_path: Path):
    """Save transcription results to JSON."""
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log.info(f"Results saved → {out_path}")


def print_summary(results: list):
    """Print a WER/latency summary table."""
    ok       = [r for r in results if "error" not in r]
    errors   = [r for r in results if "error"     in r]
    wer_vals = [r["wer"] for r in ok if "wer" in r]

    sep = "═" * 62
    print(f"\n{sep}")
    print("  ASR EVALUATION SUMMARY  (Step 1)")
    print(sep)
    print(f"  Total files    : {len(results)}")
    print(f"  Successful     : {len(ok)}")
    print(f"  Errors         : {len(errors)}")
    if ok:
        avg_lat  = sum(r["latency_s"]  for r in ok) / len(ok)
        avg_conf = sum(r["confidence"] for r in ok) / len(ok)
        print(f"  Avg latency    : {avg_lat:.2f}s")
        print(f"  Avg confidence : {avg_conf:.2%}")
    if wer_vals:
        avg_wer = sum(wer_vals) / len(wer_vals)
        print(f"  Files w/ WER   : {len(wer_vals)}")
        print(f"  Avg WER        : {avg_wer:.2%}")
        print(f"  Best WER       : {min(wer_vals):.2%}")
        print(f"  Worst WER      : {max(wer_vals):.2%}")

    if errors:
        print(f"\n  Failed files:")
        for r in errors:
            print(f"    ✗ {r['file']}: {r['error']}")
    print(f"{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 1.6  Live microphone  (optional — needs PyAudio)
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_microphone(model, duration_hint: int = None) -> str:
    """
    Record from default microphone. If duration_hint is provided, listen for that many seconds.
    Otherwise, listen until Ctrl-C.
    Requires: pip install pyaudio
    """
    try:
        import pyaudio
    except ImportError:
        raise RuntimeError(
            "PyAudio not installed.\n"
        )
    from vosk import KaldiRecognizer

    pa  = pyaudio.PyAudio()
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    rec.SetWords(True)

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=4096,
    )

    log.info("🎙  Listening…")
    parts = []
    start_time = time.time()
    
    try:
        while True:
            data = stream.read(4096, exception_on_overflow=False)
            if rec.AcceptWaveform(data):
                text = json.loads(rec.Result()).get("text", "")
                if text:
                    parts.append(text)
                    log.info(f"  Finalizing utterance: {text}")
                    break  # Stop listening automatically when a complete sentence is detected (natural pause)
            else:
                partial = json.loads(rec.PartialResult()).get("partial", "")
                if partial:
                    log.info(f"  Partial: {partial}")
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    final = json.loads(rec.FinalResult()).get("text", "")
    if final:
        parts.append(final)

    return " ".join(parts).strip()


# ─────────────────────────────────────────────────────────────────────────────
# 1.7  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step 1 — Vosk Nepali ASR Module")
    parser.add_argument("--download",   action="store_true",
                        help="Download Vosk Nepali model (run once, needs internet)")
    parser.add_argument("--audio-dir",  default=str(AUDIO_DIR),
                        help=f"Directory with audio files  (default: {AUDIO_DIR})")
    parser.add_argument("--references", default=None,
                        help="JSON file mapping filename→reference_transcript (for WER)")
    parser.add_argument("--mic",        action="store_true",
                        help="Record from microphone instead of batch files")
    parser.add_argument("--output",     default=str(OUTPUTS_DIR / "asr_results.json"),
                        help="Where to save JSON results")
    args = parser.parse_args()

    if args.download:
        success = download_model()
        sys.exit(0 if success else 1)

    model = load_model()

    if args.mic:
        text = transcribe_microphone(model)
        print(f"\nTranscript: {text}")
        return

    references = None
    if args.references:
        with open(args.references, "r", encoding="utf-8") as f:
            references = json.load(f)
        log.info(f"Loaded {len(references)} reference transcripts")

    results = batch_transcribe(Path(args.audio_dir), model, references)

    if results:
        print_summary(results)
        save_results(results, Path(args.output))
    else:
        log.warning("No results to save.")


if __name__ == "__main__":
    main()