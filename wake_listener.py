from __future__ import annotations

import collections
import math
import os
import queue
import signal
import struct
import sys
import threading
import time
from typing import Any, Dict, Generator, Tuple

import pvcobra
import pvporcupine
from dotenv import load_dotenv
from pvrecorder import PvRecorder
from stt_faster_whisper import transcribe_bytes

load_dotenv()
ACCESS_KEY = os.getenv("PORCUPINE_ACCESS_KEY")

KEYWORD_FILE_PATH = "/home/colden/Projects/Assistant/pax-ton_en_raspberry-pi_v3_0_0.ppn"

# VAD Tunables
VAD_THRESH = 0.5                  # Cobra probability threshold
CHUNK_SIL_FRAMES = 6             # ~0.19s of silence -> cut a chunk
TRAIL_SIL_FRAMES = 30             # 36 frames equivalent to ~1.2s of silence -> end utterance ( +0.5s from old value)

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


# def output_dBFS(pcm, meter_every, n):
#     lvl = peak_dbfs(pcm)
    # if n % meter_every == 0:
    #     print(f"level ~ {lvl:5.1f} dBFS", flush=True)


def listen_for_utterances(device_index: int | None = None) -> Generator[Tuple[str, Dict[str, Any], PhaseTimer | None], None, None]:
    porcupine = pvporcupine.create(
        access_key=ACCESS_KEY,
        keyword_paths=[KEYWORD_FILE_PATH]
    )
    cobra = pvcobra.create(access_key=ACCESS_KEY)

    recorder = PvRecorder(device_index=device_index, frame_length=porcupine.frame_length)

    sr = porcupine.sample_rate              # 16000

    # buffers/state
    chunk_audio = bytearray() # actual audio to be processed
    chunk_results: list[Dict[str, Any]] = [] # list of text for each speech chunk processed
    chunk_queue: queue.Queue[bytes | None] | None = None # work queue for each chunk that needs to be processed
    chunk_worker: threading.Thread | None = None # starts with target as _chunk_worker to process

    state = "IDLE"
    trailing_sil = 0              # for utterance end
    chunk_sil = 0                 # for chunk boundary
    silence_chunked = True       # track whether we've already chunked current silence run
    phase_timer: PhaseTimer | None = None

    def _chunk_worker():
        nonlocal chunk_results
        idx = 0
        while True:
            buf = chunk_queue.get()  # type: ignore[arg-type]
            if buf is None:
                chunk_queue.task_done()  # type: ignore[union-attr]
                break
            t0 = time.perf_counter()
            dur_ms = len(buf) / (sr * 2) * 1000.0
            text, meta = transcribe_bytes(
                buf,
                sample_rate_hz=sr,
                language="en",
                beam_size=3, # TODO: Adjust
                temperature=0.0,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            idx += 1
            print(f"[STT] chunk #{idx} decoded in {elapsed_ms:.1f} ms: {text!r}, chunk dur_ms: {dur_ms:.1f}", flush=True)
            chunk_results.append(dict(text=text, meta=meta, elapsed_ms=elapsed_ms, duration_ms=dur_ms))
            chunk_queue.task_done()  # type: ignore[union-attr]

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
    # frames_per_sec = max(1, int(porcupine.sample_rate / porcupine.frame_length))  # ~31 @ 16k/512
    # meter_every = frames_per_sec * 5              # ~1 second
    # n = 0

    while True:
        pcm = recorder.read()  # list[int], length == frame_len. These are the audio chunks that are processed

        # n += 1
        # output_dBFS(pcm, meter_every, n)

        frame = frame_bytes(pcm)

        if state == "IDLE":
            idx = porcupine.process(pcm)                  # -1 none; 0 => wakeword
            if idx == 0:
                print("paxton detected, wake")
                phase_timer = PhaseTimer()
                phase_timer.checkpoint("Wake word detected")
                chunk_queue = queue.Queue()
                chunk_results = []
                chunk_sil = 0
                silence_chunked = True # we initialize this to True so that if there is quiet after saying the wakeword it will not append
                chunk_worker = threading.Thread(target=_chunk_worker, daemon=True)
                chunk_worker.start()
                phase_timer.checkpoint("Started streaming STT")
                trailing_sil = 0
                state = "LISTENING"
        else:
            prob = cobra.process(pcm)                     # 0..1 voice probability
            if prob >= VAD_THRESH:
                trailing_sil = 0
                chunk_sil = 0
                silence_chunked = False
                if chunk_queue is not None:
                    chunk_audio.extend(frame) # if no silence, append to current chunk_audio
            else: # if silent, either end utterance or send chunk off for STT processing
                trailing_sil += 1
                chunk_sil += 1
                if not silence_chunked and chunk_queue is not None:
                    chunk_audio.extend(frame) # only append first silence run. This should be done to give some trailing silence chunks for whisper to listen to
                if chunk_sil >= CHUNK_SIL_FRAMES and chunk_queue is not None and chunk_audio: # put into processing queue if enough silence for chunk boundary
                    chunk_queue.put(bytes(chunk_audio))
                    chunk_audio.clear()
                    chunk_sil = 0
                    silence_chunked = True
                if trailing_sil >= TRAIL_SIL_FRAMES: # if enough trailing silence, end utterance
                    print("end utterance")
                    if phase_timer:
                        phase_timer.checkpoint("Utterance captured, finalizing STT")
                    if chunk_queue is not None:
                        chunk_queue.put(None)
                        chunk_queue.join()
                    if chunk_worker is not None:
                        chunk_worker.join()
                    text = " ".join(c["text"] for c in chunk_results).strip()
                    meta = {"chunks": chunk_results}
                    yield text, meta, phase_timer
                    state = "IDLE"
                    phase_timer = None
                    trailing_sil = 0
                    chunk_sil = 0
                    silence_chunked = True
                    chunk_audio = bytearray()
                    chunk_results = []
                    chunk_queue = None
                    chunk_worker = None
