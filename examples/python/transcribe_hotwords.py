"""Compare default transcription, hotword hints and the next default request."""

from pandora_audio8_asr_0_1b import ASRModel

MODEL_DIR = "models"
AUDIO = "examples/data/lecture_10s.wav"
HOTWORDS = ["developer experience", "open"]

with ASRModel(MODEL_DIR) as model:
    baseline = model.transcribe(AUDIO)
    print("Default:", baseline.text)

    hinted = model.transcribe(AUDIO, hotwords=HOTWORDS)
    print("Hotwords:", HOTWORDS)
    print("With hints:", hinted.text)

    following = model.transcribe(AUDIO)
    print("Default after hints:", following.text)
