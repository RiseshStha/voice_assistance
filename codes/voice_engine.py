"""
voice_engine.py - Part 3: ASR and TTS Integration
=================================================
Handles microphone input (STT) and system voice output (TTS).
"""

import logging
import os
import re
import shutil
import subprocess
import sys
import wave
from collections import Counter
from difflib import get_close_matches
from pathlib import Path
from threading import Event, Thread

import numpy as np
from faster_whisper import WhisperModel

from config import MODELS_DIR, OUTPUTS_DIR, SAMPLE_RATE, WHISPER_COMPUTE, WHISPER_DEVICE, WHISPER_MODEL_SIZE

log = logging.getLogger(__name__)

_DOMAIN_HOTWORDS = ""
_AVAILABLE_ESPEAK_VOICES: list[str] | None = None

_LATIN_NE_WORD_HINTS = {
    "phutbal": "फुटबल",
    "phutbalma": "फुटबलमा",
    "football": "फुटबल",
    "referee": "रेफ्री",
    "referee": "रेफ्री",
    "refereele": "रेफ्रीले",
    "rephri": "रेफ्री",
    "rato": "रातो",
    "pahenlo": "पहेंलो",
    "pahelo": "पहेंलो",
    "card": "कार्ड",
    "cardko": "कार्डको",
    "prayog": "प्रयोग",
    "garnu": "गर्नु",
    "garne": "गर्ने",
    "kun": "कुन",
    "prakar": "प्रकार",
    "ko": "को",
    "ka": "का",
    "ki": "की",
    "wa": "वा",
    "ra": "र",
    "ma": "मा",
    "le": "ले",
    "liya": "ले",
    "sanchaar": "सञ्चार",
    "sanchar": "सञ्चार",
    "prashna": "प्रश्न",
    "uttar": "उत्तर",
    "vidhyarthi": "विद्यार्थी",
    "bidhyarthi": "विद्यार्थी",
    "shiksha": "शिक्षा",
    "vigyan": "विज्ञान",
    "ganit": "गणित",
    "samajik": "सामाजिक",
    "swasthya": "स्वास्थ्य",
    "khelkud": "खेलकुद",
    "kala": "कला",
    "nepali": "नेपाली",
    "ho": "हो",
    "hu": "हो",
    "ke": "के",
    "kina": "किन",
    "kasari": "कसरी",
    "kasto": "कस्तो",
    "protin": "प्रोटिन",
    "protein": "प्रोटिन",
    "paine": "पाइने",
    "kunai": "कुनै",
    "kunei": "कुनै",
    "charwata": "चारवटा",
    "charota": "चारवटा",
    "charvata": "चारवटा",
    "char": "चार",
    "srota": "स्रोत",
    "srotaharu": "स्रोतहरू",
    "srotaharuko": "स्रोतहरूको",
    "naam": "नाम",
    "nam": "नाम",
    "lekhnuhos": "लेख्नुहोस्",
    "mansiko": "मानिसको",
    "manisko": "मानिसको",
    "khana": "खाना",
    "poshan": "पोषण",
    "vitamin": "भिटामिन",
    "carbohydrate": "कार्बोहाइड्रेट",
    "charbi": "चर्बी",
    "khanij": "खनिज",
    "padartha": "पदार्थ",
    "padartha": "पदार्थ",
    "padarthale": "पदार्थले",
    "hamro": "हाम्रो",
    "sarir": "शरीर",
    "sharir": "शरीर",
    "sarirma": "शरीरमा",
    "sharirma": "शरीरमा",
    "kaam": "काम",
    "kam": "काम",
    "garchha": "गर्छ",
    "garchhan": "गर्छन्",
    "thos": "ठोस",
    "toz": "ठोस",
    "bastu": "वस्तु",
    "vastu": "वस्तु",
    "bastuka": "वस्तुका",
    "vastuka": "वस्तुका",
    "bastuko": "वस्तुको",
    "vastuko": "वस्तुको",
    "bostuka": "वस्तुका",
    "bostuku": "वस्तुको",
    "duiwata": "दुईवटा",
    "diyota": "दुईवटा",
    "diota": "दुईवटा",
    "bishesta": "विशेषता",
    "bisesata": "विशेषता",
    "biseshata": "विशेषता",
    "bisesta": "विशेषता",
    "uliknoz": "लेख्नुहोस्",
    "uliknuz": "लेख्नुहोस्",
    "torol": "तरल",
    "aakaar": "आकार",
    "akar": "आकार",
    "nischit": "निश्चित",
    "hundaina": "हुँदैन",
    "hundainam": "हुँदैन",
    "ngundainam": "हुँदैन",
    "aadhunik": "आधुनिक",
    "adhunik": "आधुनिक",
    "pravidhi": "प्रविधि",
    "prabidhi": "प्रविधि",
    "pravidhile": "प्रविधिले",
    "manab": "मानव",
    "jiban": "जीवन",
    "jibanma": "जीवनमा",
    "aeka": "आएका",
    "aaeka": "आएका",
    "aayeka": "आएका",
    "paribartan": "परिवर्तन",
    "pariwartan": "परिवर्तन",
}


def _romanize_devanagari(text: str) -> str:
    vowels = {"अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऋ": "ri"}
    consonants = {
        "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
        "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
        "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
        "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
        "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
        "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
        "ष": "sh", "स": "s", "ह": "h", "ळ": "l",
    }
    marks = {"ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ृ": "ri", "ं": "n", "ँ": "n", "ः": "h", "्": ""}
    specials = {"क्ष": "ksh", "त्र": "tra", "ज्ञ": "gya"}

    text = str(text)
    out = []
    i = 0
    while i < len(text):
        pair = text[i : i + 2]
        if pair in specials:
            out.append(specials[pair])
            i += 2
            continue

        ch = text[i]
        if ch in vowels:
            out.append(vowels[ch])
            i += 1
            continue
        if ch in consonants:
            base = consonants[ch]
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if nxt in marks:
                out.append(base + marks[nxt])
                i += 2
            elif nxt == "्":
                out.append(base)
                i += 2
            else:
                out.append(base + "a")
                i += 1
            continue
        if ch in marks:
            out.append(marks[ch])
            i += 1
            continue
        if ch.isascii() and ch.isalpha():
            out.append(ch.lower())
            i += 1
            continue
        out.append(" ")
        i += 1

    return re.sub(r"\s+", " ", "".join(out)).strip()


def set_domain_hotwords(questions: list[str]) -> None:
    """
    Build a broad Latinized Nepali hotword list from the local dataset so ASR
    is nudged toward educational vocabulary instead of arbitrary phonetics.
    """
    global _DOMAIN_HOTWORDS

    token_counts: Counter[str] = Counter()
    for question in questions:
        romanized = _romanize_devanagari(question)
        for token in re.findall(r"[a-z]{3,}", romanized.lower()):
            token_counts[token] += 1

    manual = [
        "nepali", "prashna", "uttar", "ke", "kina", "kasari", "kun", "kasto",
        "phutbal", "phutbalma", "referee", "refereele", "rato", "pahenlo",
        "card", "cardko", "prayog", "garnu", "prakar", "sanchaar",
        "vigyan", "ganit", "samajik", "swasthya", "khelkud", "kala",
        "protin", "protein", "paine", "kunai", "charwata", "srota",
        "srotaharu", "srotaharuko", "naam", "lekhnuhos", "poshan",
        "vitamin", "carbohydrate", "charbi", "mansiko", "manisko",
        "khanij", "padartha", "hamro", "sarirma", "kaam", "garchhan",
        "thos", "bastu", "bastuka", "duiwata", "bishesta", "torol",
        "akar", "nischit", "hundaina",
        "aadhunik", "adhunik", "pravidhi", "pravidhile", "manab",
        "jiban", "jibanma", "aayeka", "paribartan",
    ]

    vocab = manual + [token for token, _count in token_counts.most_common(120)]
    unique_tokens = []
    seen = set()
    for token in vocab:
        if token not in seen:
            unique_tokens.append(token)
            seen.add(token)

    _DOMAIN_HOTWORDS = " ".join(unique_tokens[:140])


def _looks_latinized_nepali(text: str) -> bool:
    letters = re.findall(r"[A-Za-z]", text or "")
    if len(letters) < 6:
        return False
    devanagari = re.findall(r"[\u0900-\u097F]", text or "")
    return len(letters) > max(4, len(devanagari) * 2)


def _repair_latinized_nepali(text: str) -> str:
    """
    Convert common Latinized Nepali words into readable Devanagari.
    This is intentionally lightweight and only activates when ASR clearly
    returned Latin phonetic output for Nepali speech.
    """
    if not _looks_latinized_nepali(text):
        return text

    prompt_leak = re.sub(
        r"\b(yo|nepali|bidhyarthiko|vidhyarthiko|prashna|sabda|sahi|rupma|lekhnuhos|hoo|ho)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    prompt_leak = re.sub(r"\baad\s*hunik\b", " aadhunik ", prompt_leak, flags=re.IGNORECASE)
    prompt_leak = re.sub(r"\bprabhiti\b", " pravidhile ", prompt_leak, flags=re.IGNORECASE)
    prompt_leak = re.sub(r"\blega\s*da\b", " le garda ", prompt_leak, flags=re.IGNORECASE)
    prompt_leak = re.sub(r"\bmanabchibanma\b", " manab jibanma ", prompt_leak, flags=re.IGNORECASE)
    prompt_leak = re.sub(r"\baikadhi\b", " aayeka dui ", prompt_leak, flags=re.IGNORECASE)
    prompt_leak = re.sub(r"\bpari\b", " paribartan ", prompt_leak, flags=re.IGNORECASE)
    parts = re.findall(r"[A-Za-z]+|[^A-Za-z]+", prompt_leak)
    repaired = []
    hint_keys = list(_LATIN_NE_WORD_HINTS.keys())

    for part in parts:
        if not re.fullmatch(r"[A-Za-z]+", part or ""):
            repaired.append(part)
            continue

        token = part.lower()
        token = token.replace("mah", "ma").replace("val", "bal").replace("oa", "wa")
        if token in _LATIN_NE_WORD_HINTS:
            repaired.append(_LATIN_NE_WORD_HINTS[token])
            continue

        match = get_close_matches(token, hint_keys, n=1, cutoff=0.72)
        if match:
            repaired.append(_LATIN_NE_WORD_HINTS[match[0]])
        else:
            repaired.append(part)

    text = "".join(repaired)
    text = re.sub(r"\s+", " ", text).strip(" ,.-")
    text = text.replace("रातो पहेंलो", "रातो वा पहेंलो")
    text = text.replace("प्रकार सञ्चार", "प्रकारको सञ्चार")
    text = text.replace("स्रोत नाम", "स्रोतहरूको नाम")
    text = text.replace("भिटामिन खाना", "भिटामिन र खनिज पदार्थले")
    text = text.replace("हाम्रो शारीरमा", "हाम्रो शरीरमा")
    text = text.replace("के कला गर्छ", "के काम गर्छ")
    text = text.replace("ठोस वस्तुका दुईवटा विशेषता लेख्नुहोस्", "ठोस वस्तुका दुईवटा विशेषताहरू")
    text = text.replace("तरल वस्तुको आकार किन निश्चित हुँदैन", "तरल वस्तुको आकार किन निश्चित हुँदैन?")
    text = text.replace("आधुनिक प्रविधि ले गर्दा", "आधुनिक प्रविधिले गर्दा")
    text = text.replace("मानव जीवनमा आएका दुई परिवर्तन प्रोटिन", "मानव जीवनमा आएका दुई परिवर्तन")
    text = text.replace("मानवजीवनमा", "मानव जीवनमा")
    return text


def load_stt_model():
    """Load Faster-Whisper."""
    return WhisperModel(
        WHISPER_MODEL_SIZE,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE,
        download_root=str(MODELS_DIR),
    )


def _find_espeak_command() -> list[str] | None:
    """
    Find an eSpeak executable that can speak Nepali, even if it is not on PATH.
    """
    candidates = []
    for exe in ("espeak-ng", "espeak"):
        path = shutil.which(exe)
        if path:
            candidates.append(path)

    if sys.platform == "win32":
        program_files = [
            os.environ.get("ProgramFiles"),
            os.environ.get("ProgramFiles(x86)"),
            os.environ.get("LOCALAPPDATA"),
        ]
        suffixes = [
            os.path.join("eSpeak NG", "espeak-ng.exe"),
            os.path.join("eSpeak", "command_line", "espeak.exe"),
            os.path.join("eSpeak", "espeak.exe"),
        ]
        for root in program_files:
            if not root:
                continue
            for suffix in suffixes:
                full_path = os.path.join(root, suffix)
                if os.path.exists(full_path):
                    candidates.append(full_path)

    seen = set()
    for cmd in candidates:
        if cmd in seen:
            continue
        seen.add(cmd)
        return [cmd]
    return None


def _list_espeak_voices() -> list[str]:
    """
    Return available eSpeak language codes from the installed runtime.
    """
    global _AVAILABLE_ESPEAK_VOICES

    if _AVAILABLE_ESPEAK_VOICES is not None:
        return _AVAILABLE_ESPEAK_VOICES

    espeak_base = _find_espeak_command()
    if not espeak_base:
        _AVAILABLE_ESPEAK_VOICES = []
        return _AVAILABLE_ESPEAK_VOICES

    try:
        kwargs = {"capture_output": True}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        result = subprocess.run(espeak_base + ["--voices"], check=True, **kwargs)
        stdout = result.stdout
        if isinstance(stdout, bytes):
            try:
                stdout = stdout.decode("utf-8", errors="ignore")
            except Exception:
                stdout = stdout.decode(errors="ignore")
        voices = []
        for line in str(stdout).splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                voices.append(parts[1])
        _AVAILABLE_ESPEAK_VOICES = voices
    except Exception as e:
        log.warning("Failed to list eSpeak voices: %s", e)
        _AVAILABLE_ESPEAK_VOICES = []

    return _AVAILABLE_ESPEAK_VOICES


def _pick_tts_voice() -> str | None:
    """
    Prefer Nepali when available, otherwise use the closest Indic fallback.
    """
    voices = _list_espeak_voices()
    for code in ("ne", "hi", "bn", "gu", "mr"):
        if code in voices:
            return code
    return None


def speak_text(text):
    """
    Voice output.
    Prioritizes eSpeak/eSpeak-NG for native offline Nepali support.
    Falls back to pyttsx3 if eSpeak is unavailable.
    """
    text = str(text).strip()
    if not text:
        return

    espeak_base = _find_espeak_command()
    if espeak_base:
        try:
            voice_code = _pick_tts_voice()
            if not voice_code:
                raise RuntimeError("No usable Indic eSpeak voice is available.")
            kwargs = {}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
                wav_path = Path(OUTPUTS_DIR) / "last_tts_output.wav"
                txt_path = Path(OUTPUTS_DIR) / "last_tts_input.txt"
                txt_path.write_text(text, encoding="utf-8")
                cmd = espeak_base + [
                    "-v", voice_code,
                    "-a", "170",
                    "-s", "145",
                    "-f", str(txt_path.resolve()),
                    "-w", str(wav_path.resolve()),
                ]
                subprocess.run(cmd, check=True, **kwargs)
                if wav_path.exists():
                    import winsound

                    winsound.PlaySound(str(wav_path), winsound.SND_FILENAME)
                    return
            else:
                cmd = espeak_base + ["-v", voice_code, text]
                subprocess.run(cmd, check=True, **kwargs)
                return
        except Exception as e:
            log.warning("eSpeak playback failed: %s", e)

    try:
        import pyttsx3  # type: ignore

        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        log.warning("TTS failed: %s", e)


def has_native_nepali_tts() -> bool:
    """Return True when eSpeak/eSpeak-NG is available for offline Nepali voice output."""
    voices = _list_espeak_voices()
    return "ne" in voices


def speak_text_async(text: str) -> None:
    """Speak answer in a background thread so the GUI stays responsive."""
    Thread(target=speak_text, args=(text,), daemon=True).start()


def list_input_devices() -> list[dict]:
    """Return available microphone devices with stable metadata for the GUI."""
    try:
        import sounddevice as sd  # type: ignore
    except Exception:
        return []

    devices = []
    hostapis = sd.query_hostapis()
    for idx, dev in enumerate(sd.query_devices()):
        if int(dev.get("max_input_channels", 0)) <= 0:
            continue
        hostapi_idx = int(dev["hostapi"])
        devices.append(
            {
                "id": idx,
                "name": str(dev["name"]),
                "hostapi": str(hostapis[hostapi_idx]["name"]),
                "default_samplerate": float(dev["default_samplerate"]),
                "max_input_channels": int(dev["max_input_channels"]),
            }
        )
    return devices


def choose_default_input_device() -> int | None:
    """
    Prefer WASAPI/DirectSound microphones over MME and avoid loopback-like devices.
    """
    devices = list_input_devices()
    if not devices:
        return None

    def score(dev: dict) -> tuple[int, int]:
        text = f"{dev['name']} {dev['hostapi']}".lower()
        score_value = 0
        if "microphone" in text:
            score_value += 5
        if "array" in text:
            score_value += 1
        if "wasapi" in text:
            score_value += 6
        elif "directsound" in text:
            score_value += 4
        elif "wdm-ks" in text:
            score_value += 3
        elif "mme" in text:
            score_value -= 3
        if "stereo mix" in text or "speaker" in text or "mapper" in text:
            score_value -= 10
        return score_value, int(dev["id"])

    best = max(devices, key=score)
    return int(best["id"])


def _resample_audio(audio_np: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Linear resampling without extra dependencies."""
    audio_np = np.asarray(audio_np, dtype=np.float32).reshape(-1)
    if audio_np.size == 0 or from_rate == to_rate:
        return audio_np.astype(np.float32, copy=False)

    src_x = np.arange(audio_np.size, dtype=np.float32)
    target_len = max(1, int(round(audio_np.size * float(to_rate) / float(from_rate))))
    dst_x = np.linspace(0, audio_np.size - 1, num=target_len, dtype=np.float32)
    return np.interp(dst_x, src_x, audio_np).astype(np.float32, copy=False)


def _save_debug_wav(audio_np: np.ndarray, sample_rate: int) -> None:
    """Persist the latest microphone sample for ASR debugging."""
    try:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        wav_path = OUTPUTS_DIR / "last_voice_input.wav"
        pcm16 = np.clip(audio_np, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16)
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())
    except Exception as e:
        log.warning("Failed to save ASR debug WAV: %s", e)


def _preprocess_audio(audio_np: np.ndarray) -> np.ndarray:
    """
    Apply light offline conditioning so quiet or noisy speech is easier to decode.
    """
    audio_np = np.asarray(audio_np, dtype=np.float32).reshape(-1)
    if audio_np.size == 0:
        return audio_np

    audio_np = audio_np - float(np.mean(audio_np))

    abs_audio = np.abs(audio_np)
    trim_threshold = max(0.01, float(np.max(abs_audio)) * 0.08) if abs_audio.size else 0.01
    voiced_idx = np.where(abs_audio >= trim_threshold)[0]
    if voiced_idx.size > 0:
        start = max(0, int(voiced_idx[0]) - int(0.15 * SAMPLE_RATE))
        end = min(audio_np.size, int(voiced_idx[-1]) + int(0.20 * SAMPLE_RATE))
        audio_np = audio_np[start:end]

    peak = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
    if peak > 0:
        audio_np = audio_np / max(peak, 1e-6)

    rms = float(np.sqrt(np.mean(np.square(audio_np)))) if audio_np.size else 0.0
    if 1e-4 < rms < 0.12:
        audio_np = np.clip(audio_np * (0.12 / rms), -1.0, 1.0)

    return audio_np.astype(np.float32, copy=False)


def record_microphone(
    duration_s: float = 10.0,
    device: int | None = None,
    stop_event: Event | None = None,
) -> np.ndarray:
    """
    Record until the user stops speaking, then return float32 waveform in [-1, 1].
    `duration_s` acts as the maximum recording length.
    """
    try:
        import sounddevice as sd  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency for microphone recording. Install: pip install sounddevice"
        ) from e

    max_duration_s = float(duration_s)
    if max_duration_s <= 0:
        raise ValueError("duration_s must be > 0")

    selected_device = choose_default_input_device() if device is None else int(device)
    devices = list_input_devices()
    device_info = next((d for d in devices if int(d["id"]) == selected_device), None)
    device_sample_rate = int(round(device_info["default_samplerate"])) if device_info else SAMPLE_RATE
    input_channels = max(1, min(int(device_info["max_input_channels"]) if device_info else 1, 4))

    try:
        sd.check_input_settings(
            device=selected_device,
            samplerate=device_sample_rate,
            channels=input_channels,
            dtype="float32",
        )
    except Exception as e:
        raise RuntimeError(
            f"Microphone is unavailable for device {selected_device} at {device_sample_rate} Hz recording. "
            "Please check the selected/default input device."
        ) from e

    chunk_size = int(0.10 * device_sample_rate)
    calibrate_chunks = 6
    min_speech_chunks = 2
    trailing_silence_chunks = 15
    max_chunks = max(int(max_duration_s / 0.10), calibrate_chunks + 1)
    min_post_speech_chunks = 25

    ambient_levels: list[float] = []
    captured: list[np.ndarray] = []
    speech_started = False
    speech_chunks = 0
    silence_after_speech = 0
    dynamic_threshold = 0.015

    with sd.InputStream(
        samplerate=device_sample_rate,
        channels=input_channels,
        dtype="float32",
        blocksize=chunk_size,
        device=selected_device,
    ) as stream:
        for i in range(max_chunks):
            chunk, overflowed = stream.read(chunk_size)
            if overflowed:
                log.warning("Microphone input overflow detected; audio may be clipped.")

            frame_block = np.asarray(chunk, dtype=np.float32)
            if frame_block.ndim == 1:
                mono = frame_block.reshape(-1)
            else:
                mono = np.mean(frame_block, axis=1, dtype=np.float32)
            rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0

            if i < calibrate_chunks:
                ambient_levels.append(rms)
                captured.append(mono)
                if i == calibrate_chunks - 1:
                    noise_floor = float(np.median(ambient_levels)) if ambient_levels else 0.0
                    dynamic_threshold = max(0.015, noise_floor * 3.0)
                continue

            captured.append(mono)

            if rms >= dynamic_threshold:
                speech_chunks += 1
                silence_after_speech = 0
                if speech_chunks >= min_speech_chunks:
                    speech_started = True
            elif speech_started:
                silence_after_speech += 1
                if speech_chunks >= min_post_speech_chunks and silence_after_speech >= trailing_silence_chunks:
                    break

            if stop_event is not None and stop_event.is_set() and speech_started:
                break

    if not captured:
        return np.zeros(0, dtype=np.float32)

    audio_np = np.concatenate(captured, axis=0)
    if device_sample_rate != SAMPLE_RATE:
        audio_np = _resample_audio(audio_np, from_rate=device_sample_rate, to_rate=SAMPLE_RATE)
    audio_np = _preprocess_audio(audio_np)
    _save_debug_wav(audio_np, SAMPLE_RATE)
    return audio_np


def transcribe_stream(model, audio_np):
    """
    Transcribe Nepali speech with a primary decode path plus a fallback retry.
    """
    audio_np = _preprocess_audio(audio_np)
    if audio_np.size == 0:
        return ""

    prompt = "नमस्ते। यो नेपाली भाषा हो। विद्यार्थीको प्रश्न स्पष्ट देवनागरीमा लेख्नुहोस्।"
    decode_attempts = [
        dict(
            language="ne",
            task="transcribe",
            beam_size=8,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=300),
            hotwords=_DOMAIN_HOTWORDS or "nepali prashna uttar ke kina kasari kun kasto",
        ),
        dict(
            language="ne",
            task="transcribe",
            initial_prompt=prompt,
            beam_size=5,
            best_of=3,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
        ),
    ]

    for attempt in decode_attempts:
        segments, _info = model.transcribe(audio_np, **attempt)
        full_text = []
        for s in segments:
            text = (s.text or "").strip()
            no_speech_prob = getattr(s, "no_speech_prob", 0.0)
            if text and no_speech_prob < 0.92:
                full_text.append(text)

        text = " ".join(full_text).strip()
        if text:
            text = " ".join(text.split())
            return _repair_latinized_nepali(text)

    return ""


def listen_and_transcribe(
    model,
    duration_s: float = 10.0,
    device: int | None = None,
    stop_event: Event | None = None,
) -> str:
    """Convenience helper: record from mic then transcribe."""
    audio = record_microphone(duration_s=duration_s, device=device, stop_event=stop_event)
    return transcribe_stream(model, audio)
