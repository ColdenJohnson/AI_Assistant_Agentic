from __future__ import annotations

import base64
import os
import subprocess
import threading
from typing import Optional, Tuple

import dashscope
from dashscope.audio.qwen_tts_realtime import (
    AudioFormat,
    QwenTtsRealtime,
    QwenTtsRealtimeCallback,
)

import tts_piper

MODEL_ID = "qwen3-tts-flash-realtime"
MODEL_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
SRC_SR = 24000  # PCM_24000HZ_MONO_16BIT


class _StreamingPlayer:
    """Minimal streaming player using the same sox+aplay pipeline as Piper."""

    def __init__(self, src_sr: int):
        sox_cmd = [
            "sox",
            "-t",
            "raw",
            "-r",
            str(src_sr),
            "-e",
            "signed-integer",
            "-b",
            "16",
            "-c",
            "1",
            "-",
            "-t",
            "wav",
            "-r",
            "48000",
            "-b",
            "16",
            "-c",
            "2",
            "-",
        ]
        aplay_cmd = ["aplay", "-D", tts_piper.APLAY_DEVICE]
        self._sox = subprocess.Popen(
            sox_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        self._aplay = subprocess.Popen(aplay_cmd, stdin=self._sox.stdout)

    def write(self, pcm: bytes):
        if self._sox.stdin:
            self._sox.stdin.write(pcm)
            self._sox.stdin.flush()

    def close(self):
        try:
            if self._sox.stdin:
                self._sox.stdin.close()
        except Exception:
            pass
        try:
            self._sox.wait(timeout=5)
        except Exception:
            pass
        try:
            self._aplay.wait(timeout=5)
        except Exception:
            pass


class QwenTtsCallback(QwenTtsRealtimeCallback):
    def __init__(self):
        self.complete_event = threading.Event()
        self._player: Optional[_StreamingPlayer] = None
        self._buf = bytearray()

    def on_open(self) -> None:
        # Player is lazy-initialized on first audio delta
        return

    def on_close(self, close_status_code, close_msg) -> None:
        self._close_player()
        self.complete_event.set()

    def on_event(self, response: dict) -> None:
        try:
            resp_type = response.get("type")
            if resp_type == "response.audio.delta":
                pcm = base64.b64decode(response["delta"])
                self._buf.extend(pcm)
                if self._player is None:
                    self._player = _StreamingPlayer(SRC_SR)
                self._player.write(pcm)
            elif resp_type in ("response.done", "session.finished"):
                self._close_player()
                self.complete_event.set()
        except Exception as exc:
            print(f"[Qwen TTS error] {exc}", flush=True)
            self._close_player()
            self.complete_event.set()

    def wait_for_finished(self):
        self.complete_event.wait()

    def buffer(self) -> bytes:
        return bytes(self._buf)

    def _close_player(self):
        if self._player:
            self._player.close()
            self._player = None


def _init_api_key():
    dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "YOUR_API_KEY")


class QwenStreamingTtsSession:
    def __init__(self, voice: str = "Cherry"):
        _init_api_key()
        self._callback = QwenTtsCallback()
        self._client = QwenTtsRealtime(
            model=MODEL_ID,
            callback=self._callback,
            url=MODEL_URL,
        )
        self._client.connect()
        self._client.update_session(
            voice=voice,
            response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
            mode="server_commit",
        )

    def append_text(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        self._client.append_text(cleaned)

    def finish(self) -> bytes:
        self._client.finish()
        self._callback.wait_for_finished()
        return self._callback.buffer()

    def close(self):
        try:
            self._client.close()
        except Exception:
            pass


def _synthesize_raw(text: str) -> bytes:
    """Mirror Piper API: return PCM bytes; playback occurs during streaming."""
    cleaned = text.strip()
    if not cleaned:
        return b""

    session = QwenStreamingTtsSession()
    try:
        session.append_text(cleaned)
        return session.finish()
    finally:
        session.close()


def _play_pcm_resampled(raw_pcm: bytes, src_sr: int):
    """Playback already handled during streaming; keep for API compatibility."""
    return
