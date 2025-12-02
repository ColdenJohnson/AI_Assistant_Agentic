from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from faster_whisper import WhisperModel

# Model settings tuned for Raspberry Pi 5 CPU
_MODEL_NAME = "tiny.en" # https://whisper-api.com/blog/models/
_DEVICE = "cpu"
_COMPUTE = "int8" # int8 is fastest on CPU
_THREADS = 4 # tune: 4–6 on Pi 5

# Singleton model (load once)
_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(
            _MODEL_NAME,
            device=_DEVICE,
            compute_type=_COMPUTE,
            cpu_threads=_THREADS,
            num_workers=_THREADS,
        )
    return _model


def _pcm16le_bytes_to_float32_mono(audio_bytes: bytes) -> np.ndarray:
    # Input is already 16 kHz, mono, 16-bit little-endian PCM
    pcm_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    return pcm_int16.astype(np.float32) / 32768.0


def transcribe_bytes(
    audio_bytes: bytes,
    sample_rate_hz: int = 16000,
    *,
    language: str = "en",
    beam_size: int = 1, # TODO: tune 1, 2, 3...
    temperature: float = 0.0,
    vad_filter: bool = False,             # you already use Cobra; leave off
    no_speech_threshold: float = 0.6,     # guardrails for non-speech
    condition_on_previous_text: bool = False,
    word_timestamps: bool = False,
    return_segments: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """
    Transcribe a single VAD-delimited chunk.
    Timestamps are disabled for speed; callers simply concatenate chunk text.
    """
    audio = _pcm16le_bytes_to_float32_mono(audio_bytes)

    model = _get_model()
    segments, info = model.transcribe(
        audio,
        language=language,
        beam_size=beam_size,
        temperature=temperature,
        vad_filter=vad_filter,
        condition_on_previous_text=condition_on_previous_text,
        word_timestamps=word_timestamps,
        without_timestamps=True,
    )

    text = "".join(seg.text for seg in segments).strip()
    meta: Dict[str, Any] = {
        "language": info.language if info else None,
        "language_prob": info.language_probability if info else None,
        "duration_ms": len(audio_bytes) / (sample_rate_hz * 2.0) * 1000.0,
    }
    return text, meta
