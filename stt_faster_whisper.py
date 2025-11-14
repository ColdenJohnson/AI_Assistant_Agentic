# ~/Projects/Assistant/stt_faster_whisper.py
from __future__ import annotations
import numpy as np
from faster_whisper import WhisperModel
from typing import Optional, Dict, Any, Tuple

# TODO: can play around with both _MODEL_NAME for increased acurracy and beam size (which i reduced to min.) The pi does not have great processing power so teh current state may be teh best for now.
# https://github.com/SYSTRAN/faster-whisper?tab=readme-ov-file

# Choose a model that fits Pi 5 CPU:
# - "small.en" (good balance)
# - "distil-large-v3" (higher quality; heavier)
# Start with int8 on CPU.
_MODEL_NAME = "tiny.en"          # https://whisper-api.com/blog/models/
_DEVICE = "cpu"                   # TODO: what if ran on GPU? likely much quicker
_COMPUTE = "int8"                 # int8 is fastest on CPU
_THREADS = 4                      # tune: 4–6 on Pi 5

# Singleton model (load once)
_model: Optional[WhisperModel] = None

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
    # Your pipeline already ensures 16 kHz, mono, 16-bit little-endian PCM
    pcm_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    return pcm_int16.astype(np.float32) / 32768.0  # [-1, 1]

def transcribe_bytes(
    audio_bytes: bytes,
    sample_rate_hz: int = 16000,
    *,
    language: str = "en",
    beam_size: int = 1, # 1, 2, 3
    temperature: float = 0.0,
    vad_filter: bool = False,             # you already use Cobra; leave off
    no_speech_threshold: float = 0.6,     # guardrails for non-speech
    condition_on_previous_text: bool = False,
    word_timestamps: bool = False,
    return_segments: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """Transcribe a single utterance (already segmented by VAD)."""
    x = _pcm16le_bytes_to_float32_mono(audio_bytes)

    model = _get_model()
    segments, info = model.transcribe(
        x,
        language=language,
        beam_size=beam_size,
        temperature=temperature,
        vad_filter=vad_filter,
        no_speech_threshold=no_speech_threshold,
        condition_on_previous_text=condition_on_previous_text,
        word_timestamps=word_timestamps,
    )
    if return_segments:
        segs = [dict(start=s.start, end=s.end, text=s.text,
                     words=[dict(start=w.start, end=w.end, word=w.word) for w in (s.words or [])])
                for s in segments]
        return (" ".join(s["text"] for s in segs).strip(),
                dict(language=info.language,
                     language_prob=info.language_probability,
                     segments=segs))
    else:
        text = "".join(s.text for s in segments).strip()
        return (text,
                dict(language=info.language,
                     language_prob=info.language_probability))
