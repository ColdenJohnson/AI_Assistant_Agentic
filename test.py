import itertools, wave, numpy as np
import wake_listener
from orchestrator import handle_utterance

AUDIO = "mormonbibletest_transformed.wav"
FRAME_LEN = 512  # porcupine uses 512 @16k

with wave.open(AUDIO, "rb") as w:
    sr = w.getframerate()
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if w.getnchannels() > 1:
        pcm = pcm[::w.getnchannels()]  # mono (left)
    if sr != 16000:  # quick resample to match pipeline expectations
        target_len = int(len(pcm) * 16000 / sr)
        pcm = np.interp(np.linspace(0, len(pcm), target_len, endpoint=False), np.arange(len(pcm)), pcm).astype(np.int16)

frames = [pcm[i:i + FRAME_LEN].tolist() for i in range(0, len(pcm), FRAME_LEN)]
frames_iter = itertools.chain(frames, itertools.repeat([0] * FRAME_LEN))  # trailing silence forever

class FileRecorder:
    def __init__(self, device_index=None, frame_length=FRAME_LEN):
        self.frames = frames_iter
    def start(self): pass
    def read(self): return next(self.frames)
    def stop(self): pass
    def delete(self): pass

class DummyPorcupine:
    def __init__(self):
        self.frame_length = FRAME_LEN
        self.sample_rate = 16000
        self._tripped = False
    def process(self, _pcm):
        if self._tripped:
            return -1
        self._tripped = True
        return 0  # fire wake word immediately
    def delete(self): pass

class DummyCobra:
    def process(self, pcm):
        return 1.0 if max((abs(x) for x in pcm), default=0) > 500 else 0.0  # tiny VAD so we chunk on quiet parts
    def delete(self): pass

wake_listener.PvRecorder = FileRecorder
wake_listener.pvporcupine.create = lambda access_key=None, keyword_paths=None: DummyPorcupine()
wake_listener.pvcobra.create = lambda access_key=None: DummyCobra()

for text, meta, timer in wake_listener.listen_for_utterances():
    handle_utterance(text, meta, timer)
    break
