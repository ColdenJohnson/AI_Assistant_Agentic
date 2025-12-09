from __future__ import annotations

import atexit
import os
import queue
import threading
from typing import Any, Dict, Optional

from llm_client_langchain import stream_chat # TODO: This could eventually be turned into a boolean to use either llm_client_openrouter or llm_client_langchain. Alternatively, should roll everything into langchain and deprecate the other one.
from stt_qwen_dashscope import run_qwen_asr_loop
from wake_listener import PhaseTimer, listen_for_utterances
import tts_piper
import tts_qwen_dashscope
from tts_qwen_dashscope import QwenStreamingTtsSession
import stt_faster_whisper

DEVICE_INDEX = 0
USE_QWEN_ASR_STT = os.getenv("USE_QWEN_ASR_STT", "false").lower() == "true"
USE_QWEN_TTS = os.getenv("USE_QWEN_TTS", "false").lower() == "true"

class StreamingSpeaker:
    """Synthesize and play TTS concurrently: next chunk starts rendering while current plays."""

    def __init__(self):
        self._tts = tts_qwen_dashscope if USE_QWEN_TTS else tts_piper
        self._text_queue: queue.Queue[str | None] = queue.Queue()
        self._pcm_queue: queue.Queue[bytes | None] = queue.Queue() # pcm stands for pulse control modulation (audio)
        self._stop = threading.Event()
        self._synth_worker = threading.Thread(target=self._synth_loop, daemon=True)
        self._play_worker = threading.Thread(target=self._play_loop, daemon=True)
        self._synth_worker.start()
        self._play_worker.start()
        atexit.register(self.close)

    def _synth_loop(self):
        while True:
            text = self._text_queue.get()
            if text is None:
                self._text_queue.task_done()
                self._pcm_queue.put(None)
                break
            try:
                pcm = self._tts._synthesize_raw(text)
                self._pcm_queue.put(pcm)
            finally:
                self._text_queue.task_done()

    def _play_loop(self):
        while True:
            pcm = self._pcm_queue.get()
            if pcm is None:
                self._pcm_queue.task_done()
                break
            try:
                self._tts._play_pcm_resampled(pcm, self._tts.SRC_SR)
            finally:
                self._pcm_queue.task_done() 

    def speak(self, text: str):
        cleaned = text.strip()
        if cleaned:
            self._text_queue.put(cleaned)

    def wait_until_idle(self):
        self._text_queue.join()
        self._pcm_queue.join()

    def close(self):
        if self._stop.is_set():
            return
        self._stop.set()
        self._text_queue.put(None)
        self._synth_worker.join()
        self._play_worker.join()


_speaker = StreamingSpeaker()

''' Sends the actual call to teh LLM and streams back the response'''
def handle_llm(text: str, phase_timer: PhaseTimer | None = None):
    msgs = [
        {"role": "system", "content": "You are a home assistant. Be concise."}, # TODO: Much better prompt, pass in a prompt file
        {"role": "user", "content": text},
    ]

    phase_timer.checkpoint(f"Sending STT to LLM: {text!r}")

    _speaker.wait_until_idle()  # avoid overlapping with prior utterance
    resp: list[str] = []
    sentence: list[str] = []
    started_stream = False
    first_chunk_logged = False
    first_token_logged = False
    chunk_counter = 0
    MIN_FIRST_CHARS = 1      # say first words quickly
    MAX_FIRST_CHARS = 1
    MIN_CHARS = 1            # afterwards keep sentences longer
    MAX_CHARS = 160           # hard stop to avoid huge chunks
    tts_session: Optional[QwenStreamingTtsSession] = None

    # TODO: update flush_sentence to be more immediate. Due to new STT, can now immediately add to queue as things come in.
    # TODO: Stream text TTS instead of 1 sentence at a time (make sure that it's working as a stream instead of otherwise)
    def flush_sentence(force: bool = False):
        nonlocal started_stream, chunk_counter, first_chunk_logged, tts_session
        chunk = "".join(sentence).strip()
        if not chunk:
            return
        if force or chunk[-1:] in (".", "!", "?", "\n"): # TODO: sentences are still being flushed in chunks: should be flushed smaller than taht? I believe the queue can handle just appending words to it dierctly as they become available (verify)
            chunk_counter += 1
            label = "first" if chunk_counter == 1 else "next"
            phase_timer.checkpoint(f"\n[LLM->TTS] sending {label} chunk #{chunk_counter}: {chunk!r}")
            if USE_QWEN_TTS:
                if tts_session is None:
                    tts_session = QwenStreamingTtsSession()
                tts_session.append_text(chunk)
            else:
                _speaker.speak(chunk)
            if phase_timer:
                msg = "First TTS chunk queued" if chunk_counter == 1 and not first_chunk_logged else f"TTS chunk #{chunk_counter} queued"
                phase_timer.checkpoint(msg)
                if chunk_counter == 1:
                    first_chunk_logged = True
            sentence.clear()
            if chunk_counter == 1:
                started_stream = True

    for token in stream_chat(msgs, usage=False):
        print("\nLLM token:", flush=True)
        print(token, end="", flush=True)
        if phase_timer and not first_token_logged:
            phase_timer.checkpoint("Received first LLM token")
            first_token_logged = True
        resp.append(token) # TODO: This will become important for tool calls, as the entire response will be needed to call a tool
        sentence.append(token)
        current_len = len("".join(sentence))
        punct = any(token.endswith(p) for p in (".", "!", "?", "\n"))
        whitespace = token.endswith(" ")

        flush_sentence(force=True) # READ NOTE: This currently immediately flushes every token as it comes in. This is intended for qwen3-realtime-TTS to handle it better as a single stream (not as chunks)
        # if not started_stream:
        #     if punct or (whitespace and current_len >= MIN_FIRST_CHARS) or (whitespace and current_len >= MAX_FIRST_CHARS):
        #         flush_sentence(force=True)
        # else:
        #     if punct and current_len >= MIN_FIRST_CHARS:
        #         flush_sentence(force=True)
        #     elif whitespace and current_len >= MIN_CHARS:
        #         flush_sentence(force=True)
        #     elif current_len >= MAX_CHARS:
        #         flush_sentence(force=True)
    print()
    flush_sentence(force=True)
    if tts_session is not None:
        tts_session.finish()
        tts_session = None
    _speaker.wait_until_idle()
    if phase_timer:
        phase_timer.stop()


def handle_utterance(text: str, _meta: Dict[str, Any], phase_timer: PhaseTimer | None = None):
    print(f"STT: {text}")
    phase_timer.checkpoint(f"STT transcription complete. Sending text to LLM: {text!r}")
    handle_llm(text, phase_timer=phase_timer)


def main(device_index: int | None = None):
    idx = DEVICE_INDEX if device_index is None else device_index
    if USE_QWEN_ASR_STT:
        run_qwen_asr_loop(handle_utterance)
    else:
        stt_faster_whisper._get_model()  # warm STT model once at startup to avoid first-chunk latency -saves about 900 ms
        for text, meta, phase_timer in listen_for_utterances(device_index=idx):
            handle_utterance(text, meta, phase_timer)


if __name__ == "__main__":
    main(device_index=DEVICE_INDEX)
