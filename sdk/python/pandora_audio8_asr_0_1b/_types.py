"""Public task data and errors for the independently delivered ASR model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


class ASRError(RuntimeError):
    """Model/runtime failure; native_code retains this SDK's original identity."""
    def __init__(self, message, *, native_code=None):
        super().__init__(message)
        self.native_code = native_code


class ASRLoadError(ASRError):
    pass


class ASRInferenceError(ASRError):
    pass


class ASRStateError(ASRError):
    pass


class ASRUnsupportedError(ASRError):
    pass


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: Optional[str]
    duration_s: float
    finish_reason: Literal["stop", "length"]

    @property
    def is_complete(self):
        return self.finish_reason == "stop"


def generation_limit(value, *, default, maximum=None):
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_new_tokens must be a positive integer or None")
    if value < 1 or (maximum is not None and value > maximum):
        raise ValueError("max_new_tokens is outside this model's supported range")
    return value
