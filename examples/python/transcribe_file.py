"""Run from the repository root with an ordinarily installed SDK."""

from pandora_audio8_asr_0_1b import ASRModel

MODEL_DIR = "models"
AUDIO = "examples/data/lecture_10s.wav"

with ASRModel(MODEL_DIR) as model:
    result = model.transcribe(AUDIO)
    print(result.text, result.finish_reason)
