"""Pass real floating PCM and its source sample rate to the installed SDK."""

import soundfile as sf
from pandora_audio8_asr_0_1b import ASRModel

MODEL_DIR = "models"
AUDIO = "examples/data/lecture_10s.wav"

pcm, sample_rate = sf.read(AUDIO, dtype="float32")
with ASRModel(MODEL_DIR) as model:
    result = model.transcribe(pcm, sample_rate=sample_rate)
    print(result.text, result.finish_reason)
