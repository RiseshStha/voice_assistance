"""
STEP 5 — Full Pipeline Integration
=====================================
Connects all four stages into one real-time, fully offline system:

  Microphone → Vosk ASR → TF-IDF+SVM Intent Classifier
       → BM25 Answer Retrieval → eSpeak-NG TTS

Startup sequence
----------------
1. Load Vosk ASR model
2. Load TF-IDF + LinearSVC intent pipeline
3. Load BM25 / TF-IDF knowledge-base index
4. Loop: listen → transcribe → classify → retrieve → speak

Usage
-----
  # Interactive microphone mode
  python step5_pipeline.py

  # Text-input mode (no microphone needed — for testing)
  python step5_pipeline.py --text "भूकम्प भनेको के हो?"

  # Batch mode (reads WAV/M4A from audio dir, writes JSON)
  python step5_pipeline.py --batch

  # Benchmark: run N queries and measure latency
  python step5_pipeline.py --bench 20
"""

import sys, json, time, logging, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    AUDIO_DIR, VOSK_MODEL_DIR, SVM_MODEL_PATH,
    OUTPUTS_DIR, SAMPLE_RATE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 5.1  Load all components
# ─────────────────────────────────────────────────────────────────────────────

def load_all_components():
    """Load ASR model, intent pipeline, and KB index. Returns (model, pipe, kb)."""
    t_start = time.time()

    # ASR
    log.info("═" * 55)
    log.info("Loading ASR model …")
    from voice_assistance import load_model
    asr_model = load_model()

    # Intent classifier
    log.info("Loading intent classifier …")
    from step2_intent import load_pipeline
    svm_pipe  = load_pipeline()

    # Knowledge base
    log.info("Loading knowledge base …")
    from step3_retrieval import load_index
    kb_index  = load_index()

    log.info(f"All components loaded in {time.time()-t_start:.2f}s")
    log.info("═" * 55)
    return asr_model, svm_pipe, kb_index


# ─────────────────────────────────────────────────────────────────────────────
# 5.2  Single query through the full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_query(
    text:      str,
    svm_pipe,
    kb_index,
    speak:     bool  = True,
    verbose:   bool  = True,
) -> dict:
    """
    Run a Nepali text string through intent classification → retrieval → TTS.

    Returns a dict with all intermediate results and per-stage latencies.
    """
    result = {"query": text}
    timings = {}

    # ── Stage 2: Intent Classification ───────────────────────────────────────
    t0 = time.time()
    from step2_intent import predict_intent
    intent_result = predict_intent(text, svm_pipe)
    timings["intent_ms"]   = round((time.time() - t0) * 1000, 1)
    result["intent"]       = intent_result["intent"]
    result["intent_conf"]  = intent_result["confidence"]

    # ── Stage 3: Answer Retrieval ─────────────────────────────────────────────
    t0 = time.time()
    from step3_retrieval import retrieve
    hits = retrieve(text, kb_index, top_n=1, intent=result["intent"])
    timings["retrieval_ms"] = round((time.time() - t0) * 1000, 1)

    if hits:
        result["answer"]           = hits[0]["answer"]
        result["matched_question"] = hits[0]["question"]
        result["retrieval_score"]  = hits[0]["score"]
    else:
        result["answer"] = "माफ गर्नुहोस्, यस प्रश्नको उत्तर फेला परेन।"

    # ── Stage 4: TTS ──────────────────────────────────────────────────────────
    t0 = time.time()
    if speak:
        from step4_tts import speak as tts_speak
        tts_speak(result["answer"])
    timings["tts_ms"] = round((time.time() - t0) * 1000, 1)

    result["timings_ms"] = timings
    result["total_ms"]   = sum(timings.values())

    if verbose:
        _print_result(result)

    return result


def _print_result(r: dict):
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  Query   : {r['query']}")
    print(f"  Intent  : {r['intent']}  (conf={r['intent_conf']:.2%})")
    if "matched_question": print(f"  Matched : {r.get('matched_question','—')}")
    print(f"  Answer  : {r['answer']}")
    t = r["timings_ms"]
    print(f"\n  Timings : intent={t.get('intent_ms',0)}ms | "
          f"retrieval={t.get('retrieval_ms',0)}ms | "
          f"tts={t.get('tts_ms',0)}ms | "
          f"total={r['total_ms']}ms")
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 5.3  Microphone live loop
# ─────────────────────────────────────────────────────────────────────────────

def mic_loop(asr_model, svm_pipe, kb_index):
    """
    Continuous listen-and-respond loop using the microphone.
    Press Ctrl-C to exit.
    """
    try:
        import pyaudio
    except ImportError:
        raise RuntimeError(
            "PyAudio not installed.\n"
            "  pip install pyaudio\n"
            "  (Raspberry Pi) sudo apt install python3-pyaudio portaudio19-dev"
        )
    from vosk import KaldiRecognizer
    import json as _json

    pa  = pyaudio.PyAudio()
    rec = KaldiRecognizer(asr_model, SAMPLE_RATE)
    rec.SetWords(False)

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=4096,
    )

    print("\n🎙  Voice Assistant ready. Speak in Nepali… (Ctrl-C to stop)\n")

    try:
        while True:
            data = stream.read(4096, exception_on_overflow=False)
            if rec.AcceptWaveform(data):
                res  = _json.loads(rec.Result())
                text = res.get("text", "").strip()
                if text:
                    log.info(f"Heard: {text}")
                    run_query(text, svm_pipe, kb_index, speak=True)
    except KeyboardInterrupt:
        print("\nExiting voice assistant. Goodbye!")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


# ─────────────────────────────────────────────────────────────────────────────
# 5.4  Batch mode (WAV / M4A files)
# ─────────────────────────────────────────────────────────────────────────────

def batch_mode(asr_model, svm_pipe, kb_index):
    """Run the full pipeline on every audio file in AUDIO_DIR."""
    from voice_assistance import to_wav16k, transcribe_wav

    files = sorted(
        f for f in AUDIO_DIR.iterdir()
        if f.suffix.lower() in {".wav", ".m4a", ".mp3"}
    )
    if not files:
        log.warning(f"No audio files found in {AUDIO_DIR}")
        return []

    log.info(f"Batch mode: processing {len(files)} files …")
    all_results = []
    t_total_start = time.time()

    for src in files:
        log.info(f"\n  ── {src.name} ──")
        try:
            # ASR
            t0 = time.time()
            wav = to_wav16k(src) if src.suffix.lower() != ".wav" else src
            asr_res = transcribe_wav(wav, asr_model)
            asr_ms  = round((time.time() - t0) * 1000, 1)

            text = asr_res["transcript"]
            log.info(f"  ASR ({asr_ms}ms): {text}")

            if text.strip():
                r = run_query(text, svm_pipe, kb_index, speak=False)
                r["source_file"] = src.name
                r["asr_ms"]      = asr_ms
                r["asr_conf"]    = asr_res["confidence"]
                all_results.append(r)
            else:
                log.warning(f"  Empty transcript for {src.name}")

        except Exception as exc:
            log.error(f"  Error on {src.name}: {exc}")

    total_s = round(time.time() - t_total_start, 2)
    out = OUTPUTS_DIR / "pipeline_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    log.info(f"\nBatch done: {len(all_results)} files in {total_s}s → {out}")
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# 5.5  Quick benchmark
# ─────────────────────────────────────────────────────────────────────────────

def benchmark(svm_pipe, kb_index, n: int = 20):
    """Run N sample queries (text-only, no TTS) and report latency stats."""
    import pandas as pd
    from config import QA_EXCEL

    df = pd.read_excel(QA_EXCEL, dtype=str)
    df = df.dropna(subset=["question"])
    samples = df["question"].sample(min(n, len(df)), random_state=42).tolist()

    log.info(f"Benchmarking {len(samples)} queries (no TTS, no ASR) …")
    latencies = []
    for q in samples:
        t0 = time.time()
        run_query(q, svm_pipe, kb_index, speak=False, verbose=False)
        latencies.append((time.time() - t0) * 1000)

    import numpy as np
    lat = np.array(latencies)
    print("\n" + "═" * 50)
    print("  BENCHMARK RESULTS  (intent + retrieval only)")
    print("═" * 50)
    print(f"  Queries  : {len(latencies)}")
    print(f"  Mean     : {lat.mean():.1f} ms")
    print(f"  Median   : {np.median(lat):.1f} ms")
    print(f"  P95      : {np.percentile(lat, 95):.1f} ms")
    print(f"  Max      : {lat.max():.1f} ms")
    print("═" * 50 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 5.6  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 5 — Full Nepali Voice Assistant Pipeline"
    )
    parser.add_argument("--text",   type=str, default=None,
                        help="Run pipeline on a Nepali text string (no mic/ASR)")
    parser.add_argument("--batch",  action="store_true",
                        help="Process all audio files in AUDIO_DIR")
    parser.add_argument("--bench",  type=int, default=None,
                        metavar="N",
                        help="Benchmark with N random queries (no TTS)")
    parser.add_argument("--no-tts", action="store_true",
                        help="Disable TTS output (silent mode)")
    args = parser.parse_args()

    asr_model, svm_pipe, kb_index = load_all_components()
    speak_out = not args.no_tts

    if args.text:
        run_query(args.text, svm_pipe, kb_index, speak=speak_out)
        return

    if args.batch:
        batch_mode(asr_model, svm_pipe, kb_index)
        return

    if args.bench:
        benchmark(svm_pipe, kb_index, n=args.bench)
        return

    # Default: interactive microphone loop
    mic_loop(asr_model, svm_pipe, kb_index)


if __name__ == "__main__":
    main()
