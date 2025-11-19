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


class StreamingSpeaker:
    """Background Piper wrapper that speaks queued chunks as soon as they arrive."""

    def __init__(self):
        from tts_piper import say as piper_say  # defer heavy import until needed

        self._say = piper_say
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        atexit.register(self.close)

    def _run(self):
        while not self._stop.is_set():
            text = self._queue.get()
            if text is None:
                self._queue.task_done()
                break
            try:
                self._say(text)
            finally:
                self._queue.task_done()

    def speak(self, text: str):
        cleaned = text.strip()
        if cleaned:
            self._queue.put(cleaned)

    def wait_until_idle(self):
        self._queue.join()

    def close(self):
        if self._stop.is_set():
            return
        self._stop.set()
        self._queue.put(None)
        self._worker.join()


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
                    handle_utterance(audio_bytes, sr)     # integrate STT here
                    utterance.clear()
                    state = "IDLE"

def handle_utterance(audio_bytes: bytes, sample_rate: int):
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

        def flush_sentence(force: bool = False):
            chunk = "".join(sentence).strip()
            if not chunk:
                return
            if force or chunk[-1:] in (".", "!", "?",):
                _speaker.speak(chunk)
                sentence.clear()

        for token in stream_chat(msgs, usage=False):
            print(token, end="", flush=True)
            resp.append(token)
            sentence.append(token)
            if any(token.endswith(p) for p in (".", "!", "?", "\n")):
                flush_sentence(force=True)
            elif len("".join(sentence)) >= 80 and token.endswith(" "):
                flush_sentence(force=True)
        print()
        flush_sentence(force=True)
        _speaker.wait_until_idle()
    
    handle_llm(text)



if __name__ == "__main__":
    main(device_index=DEVICE_INDEX)
