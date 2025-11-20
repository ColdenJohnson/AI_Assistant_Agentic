import sys, signal, time, math, collections, struct, threading, queue, atexit
import pvporcupine
from pvrecorder import PvRecorder
import pvcobra
from stt_faster_whisper import transcribe_bytes
from llm_client_openrouter import stream_chat
import os
from dotenv import load_dotenv

load_dotenv()
ACCESS_KEY = os.getenv("PORCUPINE_ACCESS_KEY")

DEVICE_INDEX = 0
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
    if n % meter_every == 0:
        print(f"level ~ {lvl:5.1f} dBFS", flush=True)

def main(device_index: int | None = None):
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
                    handle_utterance(audio_bytes, sr, phase_timer)     # integrate STT here
                    utterance.clear()
                    state = "IDLE"
                    phase_timer = None

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


    # def handle_llm(text: str):
    #     msgs = [
    #         {"role": "system", "content": "You are a home assistant. Be concise."},
    #         {"role": "user", "content": text},
    #     ]
    #     for token in stream_chat(msgs, usage=False):
    #         print(token, end="", flush=True)
    #     print()
    
    # handle_llm(text)

    def handle_llm(text: str):
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
    
    handle_llm(text)



if __name__ == "__main__":
    main(device_index=DEVICE_INDEX)
