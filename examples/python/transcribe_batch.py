"""Run from the Demo root with an ordinarily installed SDK."""

from pandora_audio8_asr_0_1b import ASRModel

AUDIO_FILES = ("examples/data/lecture_10s.wav", "examples/data/lecture_tail4s.wav")

with ASRModel.from_pretrained("models") as model:
    results = model.transcribe_batch(AUDIO_FILES)

for audio, result in zip(AUDIO_FILES, results, strict=True):
    print(f"{audio}: {result.text} ({result.finish_reason})")
