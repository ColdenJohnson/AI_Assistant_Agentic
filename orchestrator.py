from __future__ import annotations

import atexit
import queue
import threading

from llm_client_openrouter import stream_chat
from stt_faster_whisper import transcribe_bytes
from wake_listener import PhaseTimer, listen_for_utterances

DEVICE_INDEX = 0


class StreamingSpeaker:
    """Synthesize and play TTS concurrently: next chunk starts rendering while current plays."""

    def __init__(self):
        import tts_piper  # defer heavy dependency until first use

        self._tts = tts_piper
        self._text_queue: queue.Queue[str | None] = queue.Queue()
        self._pcm_queue: queue.Queue[bytes | None] = queue.Queue()
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


def handle_llm(text: str, phase_timer: PhaseTimer | None = None):
    msgs = [
        {"role": "system", "content": "You are a home assistant. Be concise."},
        {"role": "user", "content": text},
    ]
    _speaker.wait_until_idle()  # avoid overlapping with prior utterance
    resp: list[str] = []
    sentence: list[str] = []
    started_stream = False
    first_chunk_logged = False
    first_token_logged = False
    chunk_counter = 0
    MIN_FIRST_CHARS = 6      # say first words quickly
    MIN_CHARS = 80            # afterwards keep sentences longer
    MAX_CHARS = 160           # hard stop to avoid huge chunks

    def flush_sentence(force: bool = False):
        nonlocal started_stream, chunk_counter, first_chunk_logged
        chunk = "".join(sentence).strip()
        if not chunk:
            return
        if force or chunk[-1:] in (".", "!", "?",):
            chunk_counter += 1
            label = "first" if chunk_counter == 1 else "next"
            print(f"\n[LLM->TTS] sending {label} chunk #{chunk_counter}: {chunk!r}")
            _speaker.speak(chunk)
            if phase_timer and chunk_counter == 1 and not first_chunk_logged:
                phase_timer.checkpoint("First TTS chunk queued")
                phase_timer.stop()
                first_chunk_logged = True
            sentence.clear()
            if chunk_counter == 1:
                started_stream = True

    for token in stream_chat(msgs, usage=False):
        print(token, end="", flush=True)
        if phase_timer and not first_token_logged:
            phase_timer.checkpoint("Received first LLM token")
            first_token_logged = True
        resp.append(token)
        sentence.append(token)
        current_len = len("".join(sentence))
        punct = any(token.endswith(p) for p in (".", "!", "?", "\n"))
        whitespace = token.endswith(" ")

        if not started_stream:
            if punct or (whitespace and current_len >= MIN_FIRST_CHARS):
                flush_sentence(force=True)
        else:
            if punct and current_len >= MIN_FIRST_CHARS:
                flush_sentence(force=True)
            elif whitespace and current_len >= MIN_CHARS:
                flush_sentence(force=True)
            elif current_len >= MAX_CHARS:
                flush_sentence(force=True)
    print()
    flush_sentence(force=True)
    _speaker.wait_until_idle()
    if phase_timer:
        phase_timer.stop()


def handle_utterance(audio_bytes: bytes, sample_rate: int, phase_timer: PhaseTimer | None = None):
    text, meta = transcribe_bytes(
        audio_bytes,
        sample_rate_hz=sample_rate,
        language="en",
        beam_size=3,
        temperature=0.0,
        vad_filter=False,                  # Cobra already used
        condition_on_previous_text=False,
        word_timestamps=False,
    )
    print(f"STT: {text}")
    if phase_timer:
        phase_timer.checkpoint("STT transcription complete")
    handle_llm(text, phase_timer=phase_timer)


def main(device_index: int | None = None):
    idx = DEVICE_INDEX if device_index is None else device_index
    for audio_bytes, sample_rate, phase_timer in listen_for_utterances(device_index=idx):
        handle_utterance(audio_bytes, sample_rate, phase_timer)


if __name__ == "__main__":
    main(device_index=DEVICE_INDEX)
