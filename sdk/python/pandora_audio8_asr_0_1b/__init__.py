"""Public Audio8 ASRModel v1 interface."""
from ._asr import ASRModel
from ._runtime import Runtime, CpuOptions, QnnOptions
from ._types import (
    ASRLoadError, ASRStateError, ASRUnsupportedError, ASRInferenceError,
    TranscriptionResult,
)

__version__ = "1.1.0"

__all__ = [
    "ASRModel", "Runtime", "CpuOptions", "QnnOptions",
    "ASRLoadError", "ASRStateError", "ASRUnsupportedError", "ASRInferenceError",
    "TranscriptionResult",
]
