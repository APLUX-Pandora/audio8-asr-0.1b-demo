"""Application-level VAD segmentation and rolling offline recognition policy."""
from collections import deque
from dataclasses import dataclass
import numpy as np

SR, FRAME = 16000, 512


class SileroVad:
    def __init__(self, path):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), np.float32)
        self.context = np.zeros(64, np.float32)

    def probability(self, frame):
        audio = np.concatenate((self.context, frame))[None]
        p, self.state = self.session.run(None, {'input': audio, 'state': self.state, 'sr': np.array(SR, np.int64)})
        self.context = frame[-64:].copy()
        return float(p.reshape(-1)[0])


@dataclass
class Request:
    samples: np.ndarray
    start: float
    end: float
    final: bool
    reason: str
    overlap: bool = False


class Segmenter:
    """Consume every input frame in order; discard only VAD-classified idle silence."""
    def __init__(self, vad, initial_s=1.5, refresh_s=.75, quiet_s=.64, max_s=24., preroll_s=.32):
        self.vad = vad
        self.initial = round(initial_s * SR)
        self.refresh = round(refresh_s * SR)
        self.quiet_limit = round(quiet_s * SR)
        self.maximum = round(max_s * SR)
        self.preroll_frames = max(1, round(preroll_s * SR / FRAME))
        self.reset()

    def reset(self):
        self.vad.reset()
        self.carry = np.zeros(0, np.float32)
        self.history = deque(maxlen=self.preroll_frames)
        self.frames = []
        self.position = self.start = self.size = self.quiet = self.last_preview = 0
        self.speaking = self.overlap = False
        self.probability = 0.

    def _request(self, final, reason):
        return Request(np.concatenate(self.frames), self.start / SR,
                       (self.start + self.size) / SR, final, reason, self.overlap)

    def push(self, samples):
        x = np.concatenate((self.carry, np.asarray(samples, np.float32)))
        n = len(x) // FRAME
        self.carry = x[n * FRAME:].copy()
        for i in range(n):
            frame = x[i * FRAME:(i + 1) * FRAME].copy()
            self.probability = self.vad.probability(frame)
            self.speaking = self.probability >= (.35 if self.speaking else .5)
            self.position += FRAME
            if not self.frames:
                self.history.append(frame)
                if not self.speaking:
                    continue
                self.frames = list(self.history)
                self.size = len(self.frames) * FRAME
                self.start = self.position - self.size
                self.history.clear()
                self.quiet = self.last_preview = 0
            else:
                self.frames.append(frame)
                self.size += FRAME
            self.quiet = 0 if self.speaking else self.quiet + FRAME
            if self.quiet >= self.quiet_limit:
                yield self._request(True, 'pause')
                self.history.extend(self.frames[-self.preroll_frames:])
                self.frames = []
                self.size = self.last_preview = self.quiet = 0
                self.overlap = False
            elif self.size >= self.maximum:
                yield self._request(True, 'window')
                # Replay a small tail to preserve acoustic context across forced cuts.
                self.frames = self.frames[-self.preroll_frames:]
                self.size = len(self.frames) * FRAME
                self.start = self.position - self.size
                self.last_preview = 0
                self.overlap = True
            elif ((not self.last_preview and self.size >= self.initial) or
                  (self.last_preview and self.size - self.last_preview >= self.refresh)):
                yield self._request(False, 'preview')
                self.last_preview = self.size

    def flush(self, reason='end'):
        if self.carry.size:
            if self.frames:
                self.frames.append(self.carry)
                self.size += len(self.carry)
            else:
                padded = np.pad(self.carry, (0, FRAME - len(self.carry)))
                if self.vad.probability(padded) >= .5:
                    self.frames = list(self.history) + [self.carry]
                    self.size = sum(map(len, self.frames))
                    self.start = self.position - sum(map(len, self.history))
            self.position += len(self.carry)
            self.carry = np.zeros(0, np.float32)
        if self.frames:
            yield self._request(True, reason)
            self.frames = []
            self.size = 0
        self.history.clear()
        self.overlap = False


def trim_overlap(previous, current):
    """Bound exact overlap trimming to the short acoustic replay, retaining uncertain text."""
    for n in range(min(8, len(previous), len(current)), 3, -1):
        prefix = current[:n]
        cjk = sum('\u3400' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' for c in prefix)
        if previous.endswith(prefix) and (len(prefix.split()) >= 2 or cjk == 4) and cjk <= 4:
            return current[n:].lstrip(), n
    return current, 0
