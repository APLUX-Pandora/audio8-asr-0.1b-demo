"""Private Web diagnostics for synchronous ASRModel calls."""
from contextlib import contextmanager
from contextvars import ContextVar

_tokens = ContextVar("audio8_web_tokens", default=None)


@contextmanager
def capture_tokens():
    result = []
    token = _tokens.set(result)
    try:
        yield result
    finally:
        _tokens.reset(token)


def record_tokens(ids):
    target = _tokens.get()
    if target is not None:
        target.append(tuple(map(int, ids)))


def load_model(model_dir, *, library=None):
    from ._asr import ASRModel
    if library is None:
        return ASRModel.from_pretrained(model_dir)
    return ASRModel._load(model_dir, library=library)
