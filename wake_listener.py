from __future__ import annotations

import collections
import math
import os
import signal
import struct
import sys
import time
from typing import Generator, Tuple

import pvcobra
import pvporcupine
from dotenv import load_dotenv
from pvrecorder import PvRecorder

load_dotenv()
ACCESS_KEY = os.getenv("PORCUPINE_ACCESS_KEY")

KEYWORD_FILE_PATH = "/home/colden/Projects/Assistant/pax-ton_en_raspberry-pi_v3_0_0.ppn"

# VAD Tunables
VAD_THRESH = 0.5          # Cobra probability threshold
TRAIL_SIL_FRAMES = 25     # ~0.8s at 16k/512
PREROLL_MS = 400


class PhaseTimer:
    """Simple helper to log latency checkpoints in milliseconds."""

    def __init__(self):
        self._start_ts: float | None = None
        self._last_ts: float | None = None
        self._active = True

    def checkpoint(self, label: str):
        if not self._active:
            return
        now = time.perf_counter()
        if self._start_ts is None:
            self._start_ts = now
            delta_ms = 0.0
            total_ms = 0.0
        else:
            delta_ms = (now - self._last_ts) * 1000.0
            total_ms = (now - self._start_ts) * 1000.0
        self._last_ts = now
        print(f"[LATENCY] {label}: +{delta_ms:.1f} ms (total {total_ms:.1f} ms)", flush=True)

    def stop(self):
        self._active = False


def frame_bytes(pcm: list[int]) -> bytes:
    # pack signed int16 list -> little-endian bytes
    return struct.pack('<{}h'.format(len(pcm)), *pcm)


def peak_dbfs(pcm):
    peak = 1
    for s in pcm:
        a = s if s >= 0 else -s
        if a > peak:
            peak = a
    return 20.0 * math.log10(peak / 32768.0)


def output_dBFS(pcm, meter_every, n):
    lvl = peak_dbfs(pcm)
    # if n % meter_every == 0:
    #     print(f"level ~ {lvl:5.1f} dBFS", flush=True)


def listen_for_utterances(device_index: int | None = None) -> Generator[Tuple[bytes, int, PhaseTimer | None], None, None]:
    porcupine = pvporcupine.create(
        access_key=ACCESS_KEY,
        keyword_paths=[KEYWORD_FILE_PATH]
    )
    cobra = pvcobra.create(access_key=ACCESS_KEY)

    recorder = PvRecorder(device_index=device_index, frame_length=porcupine.frame_length)

    # derive sizes
    frame_len = porcupine.frame_length      # e.g., 512
    sr = porcupine.sample_rate              # 16000
    preroll_frames = max(1, PREROLL_MS * sr // 1000 // frame_len)

    # buffers
    preroll = collections.deque(maxlen=preroll_frames)  # holds bytes per frame
    utterance = bytearray()

    state = "IDLE"
    trailing_sil = 0
    phase_timer: PhaseTimer | None = None

    def cleanup(*_):
        try:
            recorder.stop()
        except Exception:
            pass
        recorder.delete()
        cobra.delete()
        porcupine.delete()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    recorder.start()
    print("listening…")
    frames_per_sec = max(1, int(porcupine.sample_rate / porcupine.frame_length))  # ~31 @ 16k/512
    meter_every = frames_per_sec * 5              # ~1 second
    n = 0

    while True:
        pcm = recorder.read()  # list[int], length == frame_len

        n += 1
        output_dBFS(pcm, meter_every, n)

        if state == "IDLE":
            preroll.append(frame_bytes(pcm))              # pack and keep recent audio
            idx = porcupine.process(pcm)                  # -1 none; 0 => wakeword
            if idx == 0:
                print("paxton detected, wake")
                phase_timer = PhaseTimer()
                phase_timer.checkpoint("Wake word detected")
                utterance = bytearray().join(preroll) if preroll else bytearray()
                trailing_sil = 0
                state = "LISTENING"
        else:
            prob = cobra.process(pcm)                     # 0..1 voice probability
            utterance.extend(frame_bytes(pcm))            # append current frame bytes
            if prob >= VAD_THRESH:
                trailing_sil = 0
            else:
                trailing_sil += 1
                if trailing_sil >= TRAIL_SIL_FRAMES:
                    print("end utterance")
                    audio_bytes = bytes(utterance)        # 16kHz, mono, 16-bit PCM
                    if phase_timer:
                        phase_timer.checkpoint("Utterance captured, running STT")
                    yield audio_bytes, sr, phase_timer
                    utterance.clear()
                    state = "IDLE"
                    phase_timer = None

