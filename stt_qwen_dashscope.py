from __future__ import annotations

import base64
import os
import signal
import sys
from typing import Any, Callable, Dict

import dashscope
import pvcobra
import pvporcupine
from dashscope.audio.qwen_omni import MultiModality, OmniRealtimeCallback, OmniRealtimeConversation
from dashscope.audio.qwen_omni.omni_realtime import TranscriptionParams
from dotenv import load_dotenv
from pvrecorder import PvRecorder

from wake_listener import KEYWORD_FILE_PATH, PhaseTimer, TRAIL_SIL_FRAMES, VAD_THRESH, frame_bytes

MODEL_ID = "qwen3-asr-flash-realtime"
MODEL_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
SAMPLE_RATE = 16000
CHANNELS = 1

load_dotenv()
ACCESS_KEY = os.getenv("PORCUPINE_ACCESS_KEY")

HANDLE_UTTERANCE_FN: Callable[[str, Dict[str, Any], PhaseTimer | None], None] = lambda t, m, p: None
_conversation: OmniRealtimeConversation | None = None
_active_phase_timer: PhaseTimer | None = None
_listen_state = "IDLE"


class QwenASRCallback(OmniRealtimeCallback):
    def on_open(self) -> None:
        print("DashScope ASR connection opened.")

    def on_close(self, close_status_code, close_msg) -> None:
        print(f"DashScope ASR connection closed with code: {close_status_code}, msg: {close_msg}")

    def on_event(self, response: dict) -> None:
        global _active_phase_timer, _listen_state
        try:
            if response.get("type") == "conversation.item.input_audio_transcription.completed":
                transcript = response.get("transcript", "")
                meta: Dict[str, Any] = {"backend": MODEL_ID}
                phase_timer = _active_phase_timer
                if phase_timer:
                    phase_timer.checkpoint(f"Qwen ASR transcription complete. Sending text to LLM: {transcript!r}")
                HANDLE_UTTERANCE_FN(transcript, meta, phase_timer)
                _active_phase_timer = None
                _listen_state = "IDLE"
        except Exception as exc:
            print(f"[DashScope ASR error] {exc}", flush=True)


def _cleanup(recorder: PvRecorder | None, cobra: pvcobra.Cobra | None, porcupine: pvporcupine.Porcupine | None):
    try:
        if _conversation:
            _conversation.close()
    except Exception:
        pass
    try:
        if recorder:
            recorder.stop()
    except Exception:
        pass
    try:
        if recorder:
            recorder.delete()
    except Exception:
        pass
    try:
        if cobra:
            cobra.delete()
    except Exception:
        pass
    try:
        if porcupine:
            porcupine.delete()
    except Exception:
        pass


def run_qwen_asr_loop(handle_utterance_fn: Callable[[str, Dict[str, Any], PhaseTimer | None], None]) -> None:
    global HANDLE_UTTERANCE_FN, _conversation, _listen_state, _active_phase_timer
    HANDLE_UTTERANCE_FN = handle_utterance_fn

    dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "YOUR_API_KEY")

    porcupine = pvporcupine.create(access_key=ACCESS_KEY, keyword_paths=[KEYWORD_FILE_PATH])
    cobra = pvcobra.create(access_key=ACCESS_KEY)
    recorder = PvRecorder(frame_length=porcupine.frame_length)

    def _handle_exit(sig, frame):
        _cleanup(recorder, cobra, porcupine)
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_exit)
    signal.signal(signal.SIGTERM, _handle_exit)

    callback = QwenASRCallback()
    _conversation = OmniRealtimeConversation(
        model=MODEL_ID,
        url=MODEL_URL,
        callback=callback,
    )
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

    recorder.start()
    print("Listening for wake word and streaming to DashScope... Press Ctrl+C to stop.")

    trailing_sil = 0
    streaming_audio = False
    try:
        while True:
            pcm = recorder.read()

            if _listen_state == "IDLE":
                idx = porcupine.process(pcm)
                if idx == 0:
                    print("Wake word detected, start listening.")
                    _active_phase_timer = PhaseTimer()
                    _active_phase_timer.checkpoint("Wake word detected")
                    trailing_sil = 0
                    streaming_audio = False
                    _listen_state = "LISTENING"
                continue

            if _listen_state == "WAITING":
                continue

            prob = cobra.process(pcm)
            frame = frame_bytes(pcm)

            if prob >= VAD_THRESH:
                trailing_sil = 0
                if _active_phase_timer and not streaming_audio:
                    _active_phase_timer.checkpoint("Started streaming STT")
                    streaming_audio = True
                if _conversation:
                    audio_b64 = base64.b64encode(frame).decode("ascii")
                    _conversation.append_audio(audio_b64)
            else:
                trailing_sil += 1
                if not streaming_audio and trailing_sil >= TRAIL_SIL_FRAMES:
                    # Wake word fired but no speech followed; reset to idle.
                    _listen_state = "IDLE"
                    trailing_sil = 0
                    if _active_phase_timer:
                        _active_phase_timer.stop()
                        _active_phase_timer = None
                    continue

                if streaming_audio and _conversation:
                    audio_b64 = base64.b64encode(frame).decode("ascii")
                    if trailing_sil <= TRAIL_SIL_FRAMES:
                        _conversation.append_audio(audio_b64)
                    if trailing_sil >= TRAIL_SIL_FRAMES:
                        print("End of speech detected.")
                        if _active_phase_timer:
                            _active_phase_timer.checkpoint("Utterance captured, awaiting transcription")
                        streaming_audio = False
                        trailing_sil = 0
                        _listen_state = "WAITING"
    finally:
        _cleanup(recorder, cobra, porcupine)
