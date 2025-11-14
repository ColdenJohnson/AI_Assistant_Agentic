# tts_piper.py
# Minimal, Pi-ready Piper TTS that PLAYS audio at 48 kHz stereo via sox+aplay.
# Requirements: piper binary in PATH, sox, aplay (ALSA). Voice .onnx (+ .onnx.json)
# Env:
#   PIPER_MODEL=/path/to/voice.onnx         (required)
#   PIPER_CONFIG=/path/to/voice.onnx.json   (optional; auto-detected if alongside)
#   APLAY_DEVICE=default                    (optional; e.g., "hw:3,0")
# Usage:
#   python tts_piper.py "hello there"
# Programmatic:
#   from tts_piper import say; say("hello")

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

PIPER_BIN = shutil.which("piper") or "piper"
MODEL_PATH = "/home/colden/Projects/Assistant/en_US-lessac-medium.onnx"
CONFIG_PATH = os.environ.get("PIPER_CONFIG")
APLAY_DEVICE = "hw:0,0"

if not MODEL_PATH:
    sys.exit("Set PIPER_MODEL=/path/to/voice.onnx")

# Derive CONFIG_PATH and source sample rate
def _detect_sr():
    cfg = None
    if CONFIG_PATH and Path(CONFIG_PATH).is_file():
        cfg = CONFIG_PATH
    else:
        # try sibling .json next to the model
        guess = Path(MODEL_PATH).with_suffix(Path(MODEL_PATH).suffix + ".json")
        if guess.is_file():
            cfg = str(guess)

    if not cfg:
        # Fallback common Piper rates; 22050 is typical for many voices
        return None, 22050

    with open(cfg, "r", encoding="utf-8") as f:
        meta = json.load(f)
    sr = int(meta.get("sample_rate", 22050))
    return cfg, sr

CONFIG_PATH, SRC_SR = _detect_sr()

def _synthesize_raw(text: str) -> bytes:
    """Run Piper once and return raw int16 mono PCM bytes at model's native sample rate."""
    cmd = [PIPER_BIN, "--model", MODEL_PATH, "--output-raw"]
    if CONFIG_PATH:
        cmd += ["--config", CONFIG_PATH]

    # Piper reads text from stdin, emits raw S16_LE mono PCM to stdout
    proc = subprocess.run(
        cmd,
        input=(text.strip() + "\n").encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True
    )
    return proc.stdout  # raw PCM

def _play_pcm_resampled(raw_pcm: bytes, src_sr: int):
    """Resample to exactly 48000 Hz, stereo, 16-bit, and play via ALSA."""
    # sox: interpret stdin as raw S16_LE mono at src_sr; resample -> 48000 Hz, 2ch, 16-bit WAV on stdout
    sox_cmd = [
        "sox",
        "-t", "raw",
        "-r", str(src_sr),
        "-e", "signed-integer",
        "-b", "16",
        "-c", "1",
        "-",                   # stdin
        "-t", "wav",
        "-r", "48000",
        "-b", "16",
        "-c", "2",
        "-"                    # stdout to aplay
    ]
    aplay_cmd = ["aplay", "-D", APLAY_DEVICE]

    sox = subprocess.Popen(sox_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    aplay = subprocess.Popen(aplay_cmd, stdin=sox.stdout)

    # Feed raw PCM to sox
    sox.stdin.write(raw_pcm)
    sox.stdin.close()

    # Wait for pipeline to finish
    sox.wait()
    aplay.wait()

def say(text: str):
    """Synthesize `text` with Piper and play it at 48 kHz stereo via ALSA."""
    if not text or not text.strip():
        return
    raw = _synthesize_raw(text)
    _play_pcm_resampled(raw, SRC_SR)

if __name__ == "__main__":
    txt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Text to speech is ready."
    say(txt)
