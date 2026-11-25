"""Pace a WAV file like an audio source and transcribe complete short windows."""

import argparse
import time
import wave

import numpy as np
from pandora_audio8_asr_0_1b import ASRModel

SAMPLE_RATE = 16_000
CHUNK_FRAMES = SAMPLE_RATE // 4
WINDOW_FRAMES = 8 * SAMPLE_RATE
MIN_WINDOW_FRAMES = 201
SAMPLE_BYTES = 2


def paced_windows(source):
    started = time.monotonic()
    end_frame = 0
    start_frame = 0
    window = bytearray()
    total_frames = source.getnframes()

    while end_frame < total_frames:
        chunk_frames = min(CHUNK_FRAMES, total_frames - end_frame)
        end_frame += chunk_frames
        time.sleep(max(0, started + end_frame / SAMPLE_RATE - time.monotonic()))
        chunk = source.readframes(chunk_frames)
        if len(chunk) != chunk_frames * SAMPLE_BYTES:
            raise ValueError("WAV ended before its declared frame count")
        window.extend(chunk)
        if end_frame - start_frame == WINDOW_FRAMES or end_frame == total_frames:
            pcm = np.frombuffer(window, dtype="<i2").astype(np.float32) / 32768.0
            if pcm.size < MIN_WINDOW_FRAMES:
                pcm = np.pad(pcm, (0, MIN_WINDOW_FRAMES - pcm.size))
            yield start_frame / SAMPLE_RATE, end_frame / SAMPLE_RATE, pcm
            window.clear()
            start_frame = end_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="?", default="examples/data/lecture_10s.wav")
    parser.add_argument("--models", default="models")
    args = parser.parse_args()

    with wave.open(args.audio, "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, SAMPLE_BYTES, SAMPLE_RATE):
            parser.error("Use a mono 16 kHz PCM16 WAV file")
        if source.getnframes() == 0:
            parser.error("The WAV file contains no audio")
        with ASRModel.from_pretrained(args.models) as model:
            for start_s, end_s, pcm in paced_windows(source):
                result = model.transcribe(pcm, sample_rate=SAMPLE_RATE)
                print(f"[{start_s:6.2f}–{end_s:6.2f} s] {result.text}", flush=True)


if __name__ == "__main__":
    main()
