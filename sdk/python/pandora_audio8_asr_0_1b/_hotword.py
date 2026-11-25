"""Upstream-style hotword helper, compiled by the native tokenizer/trie core.

The returned immutable plan is accepted by this SDK's logits_processor subset;
there is no arbitrary Python callback into the native decoding loop.
"""
from dataclasses import dataclass
import json
import math
import re


@dataclass(frozen=True)
class _HotwordProcessor:
    payload: bytes
    owner_key: object


def make_hotword_processor(tokenizer, hotwords: str, *, topk: int,
                           start_boost: float, continuation_boost: float):
    # String coercion is Python input representation. Unicode trimming, variants,
    # filtering, tokenization, trie construction and bias remain native.
    if isinstance(hotwords, (list, tuple)):
        words = [str(x) for x in hotwords]
    elif hotwords:
        words = re.split(r"[,，]", str(hotwords))
    else:
        words = []
    k = int(topk)
    k = 0 if k <= 0 else min(k, 151936)
    start, continuation = float(start_boost), float(continuation_boost)
    if not all(math.isfinite(x) and abs(x) <= 3.4028234663852886e38 for x in (start, continuation)):
        raise ValueError("Boosts must be finite float32-representable values")
    if not hasattr(tokenizer, "_compile_hotwords"):
        raise TypeError("Use this SDK processor's native-backed tokenizer")
    payload = tokenizer._compile_hotwords(words, k, start, continuation)
    if not json.loads(payload)["sequences"]:
        return None
    return _HotwordProcessor(payload, tokenizer._model_key)
