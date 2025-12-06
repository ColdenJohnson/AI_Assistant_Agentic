from __future__ import annotations

import base64
import os
import signal
import sys
import time
from typing import Any, Callable, Dict

import dashscope
import pyaudio
from dashscope.audio.qwen_omni import MultiModality, OmniRealtimeCallback, OmniRealtimeConversation
from dashscope.audio.qwen_omni.omni_realtime import TranscriptionParams

from wake_listener import PhaseTimer

MODEL_ID = "qwen3-asr-flash-realtime"
MODEL_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_BYTES = 3200  # 16-bit mono @16kHz -> 1600 frames (~0.1s)
CHUNK_FRAMES = CHUNK_BYTES // 2

_pya: pyaudio.PyAudio | None = None
_mic_stream: pyaudio.Stream | None = None
_conversation: OmniRealtimeConversation | None = None
_device_index: int | None = None

HANDLE_UTTERANCE_FN: Callable[[str, Dict[str, Any], PhaseTimer | None], None] = lambda t, m, p: None


def _close_mic():
    global _pya, _mic_stream
    try:
        if _mic_stream:
            _mic_stream.stop_stream()
            _mic_stream.close()
    finally:
        _mic_stream = None
    try:
        if _pya:
            _pya.terminate()
    finally:
        _pya = None


class QwenASRCallback(OmniRealtimeCallback):
    def on_open(self) -> None:
        global _pya, _mic_stream
        print("DashScope ASR connection opened, initializing microphone...")
        _pya = pyaudio.PyAudio()
        _mic_stream = _pya.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=_device_index,
            frames_per_buffer=CHUNK_FRAMES,
        )

    def on_close(self, close_status_code, close_msg) -> None:
        print(f"DashScope ASR connection closed with code: {close_status_code}, msg: {close_msg}")
        _close_mic()

    def on_event(self, response: dict) -> None:
        try:
            if response.get("type") == "conversation.item.input_audio_transcription.completed":
                transcript = response.get("transcript", "")
                phase_timer = PhaseTimer()
                meta: Dict[str, Any] = {"backend": MODEL_ID}
                phase_timer.checkpoint(f"Qwen ASR transcription complete. Sending text to LLM: {transcript!r}")
                HANDLE_UTTERANCE_FN(transcript, meta, phase_timer)
        except Exception as exc:
            print(f"[DashScope ASR error] {exc}")


def run_qwen_asr_loop(
    handle_utterance_fn: Callable[[str, Dict[str, Any], PhaseTimer | None], None],
    device_index: int | None = None,
) -> None:
    global HANDLE_UTTERANCE_FN, _conversation, _device_index
    HANDLE_UTTERANCE_FN = handle_utterance_fn
    _device_index = device_index

    dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "YOUR_API_KEY")

    callback = QwenASRCallback()
    _conversation = OmniRealtimeConversation(
        model=MODEL_ID,
        url=MODEL_URL,
        callback=callback,
    )

    def _handle_exit(sig, frame):
        if _conversation:
            _conversation.close()
        _close_mic()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_exit)
    signal.signal(signal.SIGTERM, _handle_exit)

    _conversation.connect()

    transcription_params = TranscriptionParams(
        language="zh",
        sample_rate=SAMPLE_RATE,
        input_audio_format="pcm",
    )
    _conversation.update_session(
        output_modalities=[MultiModality.TEXT],
        enable_input_audio_transcription=True,
        transcription_params=transcription_params,
    )

    print("DashScope ASR streaming... Press Ctrl+C to stop.")
    while True:
        if _mic_stream:
            audio_data = _mic_stream.read(CHUNK_FRAMES, exception_on_overflow=False)
            audio_b64 = base64.b64encode(audio_data).decode("ascii")
            _conversation.append_audio(audio_b64)
        else:
            time.sleep(0.01)
