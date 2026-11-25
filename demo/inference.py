"""Web adaptation of ASRModel transcription and private token diagnostics."""
import os
from dataclasses import dataclass

from pandora_audio8_asr_0_1b._web import capture_tokens


@dataclass(frozen=True)
class Recognition:
    text: str
    token_ids: tuple
    finish_reason: str

    @property
    def is_complete(self):
        return self.finish_reason == "stop"


def recognize(model, audio, *, hotwords=None, max_new_tokens=256):
    settings = hotwords or {}
    words = settings.get("words") if settings.get("enabled", False) else None
    kwargs = {} if isinstance(audio, (str, os.PathLike)) else {"sample_rate": 16000}
    with capture_tokens() as diagnostics:
        result = model.transcribe(audio, hotwords=words, max_new_tokens=max_new_tokens, **kwargs)
    if len(diagnostics) != 1:
        raise RuntimeError("Missing internal transcription diagnostics")
    return Recognition(result.text, diagnostics[0], result.finish_reason)
