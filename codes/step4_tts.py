"""
STEP 4 — Text-to-Speech (TTS) Module
======================================
Offline Nepali TTS using eSpeak-NG.

Primary backend  : eSpeak-NG  (subprocess call — must be installed system-wide)
Fallback backend : pyttsx3    (pip install pyttsx3)

Installation
------------
Windows:
  Download eSpeak-NG installer from https://github.com/espeak-ng/espeak-ng/releases
  Install then ensure 'espeak-ng' is on your PATH (or set ESPEAK_PATH below).

Raspberry Pi / Linux:
  sudo apt install espeak-ng

Usage
-----
  # Speak a Nepali string aloud
  python step4_tts.py --speak "नमस्ते, तपाईंलाई कसरी सहयोग गर्न सक्छु?"

  # Save to WAV file
  python step4_tts.py --speak "नमस्ते" --save outputs/hello.wav

  # Test TTS (speaks a default sentence)
  python step4_tts.py --test
"""

import sys, subprocess, logging, argparse, shutil, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── eSpeak-NG configuration ───────────────────────────────────────────────────
# Override this if espeak-ng is not on PATH (e.g. Windows custom install)
ESPEAK_PATH = shutil.which("espeak-ng") or "espeak-ng"

NEPALI_LANG   = "ne"          # eSpeak-NG Nepali language code
DEFAULT_SPEED = 140           # words per minute (120–160 recommended for clarity)
DEFAULT_PITCH = 50            # 0–99
DEFAULT_VOLUME = 100          # %


# ─────────────────────────────────────────────────────────────────────────────
# 4.1  Backend detection
# ─────────────────────────────────────────────────────────────────────────────

def check_espeak() -> bool:
    """Return True if eSpeak-NG is available on PATH."""
    try:
        r = subprocess.run(
            [ESPEAK_PATH, "--version"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0:
            log.info(f"eSpeak-NG found: {r.stdout.strip()}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    log.warning(
        "eSpeak-NG not found on PATH.\n"
        "  Windows : download from https://github.com/espeak-ng/espeak-ng/releases\n"
        "  Linux   : sudo apt install espeak-ng"
    )
    return False


def check_pyttsx3() -> bool:
    """Return True if pyttsx3 is importable."""
    try:
        import pyttsx3
        return True
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 4.2  TTS functions
# ─────────────────────────────────────────────────────────────────────────────

def speak_espeak(
    text:    str,
    lang:    str   = NEPALI_LANG,
    speed:   int   = DEFAULT_SPEED,
    pitch:   int   = DEFAULT_PITCH,
    volume:  int   = DEFAULT_VOLUME,
    wav_out: Path  = None,
) -> bool:
    """
    Synthesise text with eSpeak-NG.

    Parameters
    ----------
    wav_out : if provided, save audio to this WAV path instead of playing aloud.

    Returns True on success.
    """
    cmd = [
        ESPEAK_PATH,
        "-v", lang,
        "-s", str(speed),
        "-p", str(pitch),
        "-a", str(volume),
    ]

    if wav_out:
        cmd += ["-w", str(wav_out), text]
    else:
        cmd.append(text)

    try:
        result = subprocess.run(cmd, capture_output=True, text=False, timeout=30)
        if result.returncode != 0:
            log.error(f"eSpeak-NG error: {result.stderr.decode(errors='ignore')}")
            return False
        if wav_out:
            log.info(f"TTS audio saved → {wav_out}")
        else:
            log.info("TTS playback complete.")
        return True
    except FileNotFoundError:
        log.error("eSpeak-NG executable not found. Is it installed and on PATH?")
        return False
    except subprocess.TimeoutExpired:
        log.error("eSpeak-NG timed out.")
        return False


def speak_pyttsx3(text: str, lang: str = NEPALI_LANG) -> bool:
    """
    Fallback TTS using pyttsx3.
    NOTE: pyttsx3 may not support Nepali on all systems; it will fall back
    to the system default voice.
    """
    try:
        import pyttsx3
        engine = pyttsx3.init()
        # Attempt to set Nepali voice if available
        voices = engine.getProperty("voices")
        for v in voices:
            if "ne" in v.id.lower() or "nepali" in v.name.lower():
                engine.setProperty("voice", v.id)
                break
        engine.setProperty("rate",   DEFAULT_SPEED * 8)
        engine.setProperty("volume", DEFAULT_VOLUME / 100)
        engine.say(text)
        engine.runAndWait()
        return True
    except Exception as exc:
        log.error(f"pyttsx3 TTS failed: {exc}")
        return False


def speak(
    text:    str,
    lang:    str  = NEPALI_LANG,
    speed:   int  = DEFAULT_SPEED,
    pitch:   int  = DEFAULT_PITCH,
    wav_out: Path = None,
) -> bool:
    """
    Main TTS entry-point.

    Tries eSpeak-NG first, falls back to pyttsx3, then prints the text
    if neither is available (graceful degradation for headless hardware).
    """
    text = text.strip()
    if not text:
        return True

    # Primary: eSpeak-NG
    if check_espeak():
        return speak_espeak(text, lang=lang, speed=speed,
                            pitch=pitch, wav_out=wav_out)

    # Fallback: pyttsx3
    if check_pyttsx3():
        log.info("Falling back to pyttsx3 TTS …")
        return speak_pyttsx3(text, lang=lang)

    # Last resort: print only (headless / no audio hardware)
    log.warning("No TTS backend available. Printing text only:")
    print(f"\n[TTS OUTPUT]: {text}\n")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# 4.3  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step 4 — Nepali TTS Module")
    parser.add_argument("--speak",  type=str, default=None,
                        help="Nepali text to synthesise")
    parser.add_argument("--save",   type=str, default=None,
                        help="Save TTS output to this WAV file instead of playing")
    parser.add_argument("--speed",  type=int, default=DEFAULT_SPEED,
                        help=f"Words per minute (default: {DEFAULT_SPEED})")
    parser.add_argument("--pitch",  type=int, default=DEFAULT_PITCH,
                        help=f"Pitch 0-99 (default: {DEFAULT_PITCH})")
    parser.add_argument("--test",   action="store_true",
                        help="Test TTS with a Nepali sample sentence")
    args = parser.parse_args()

    wav_out = Path(args.save) if args.save else None

    if args.test:
        test_text = "नमस्ते, म तपाईंको नेपाली विद्यार्थी सहायक हुँ। कसरी मद्दत गर्न सक्छु?"
        log.info(f"Test sentence: {test_text}")
        speak(test_text, speed=args.speed, pitch=args.pitch, wav_out=wav_out)
        return

    if args.speak:
        speak(args.speak, speed=args.speed, pitch=args.pitch, wav_out=wav_out)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
