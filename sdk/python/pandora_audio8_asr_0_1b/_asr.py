"""Synchronous Audio8 speech recognition over the native engine."""

import os
import threading

from ._audio import prepare_audio
from ._types import (
    ASRLoadError,
    ASRStateError,
    ASRUnsupportedError,
    TranscriptionResult,
    generation_limit,
)
from ._binding import text
from ._engine import _GenerationModel, _Processor
from ._hotword import make_hotword_processor


class ASRModel:
    """Load a model once and transcribe independent audio requests."""

    def __init__(self, model_dir, *, runtime=None):
        self._initialize(model_dir, runtime=runtime)

    @classmethod
    def from_pretrained(cls, model_dir, *, runtime=None):
        return cls(model_dir, runtime=runtime)

    @classmethod
    def _load(cls, model_dir, *, runtime=None, library=None):
        obj = cls.__new__(cls)
        obj._initialize(model_dir, runtime=runtime, library=library)
        return obj

    def _initialize(self, model_dir, *, runtime, library=None):
        self._pid = os.getpid()
        self._request = threading.Lock()
        self._closing = threading.Event()
        self._closed = False
        self._processor = self._model = None
        try:
            self._model = _GenerationModel.from_pretrained(
                model_dir, runtime=runtime, library=library
            )
            self._processor = _Processor._from_model(self._model)
        except BaseException as error:
            try:
                self.close()
            except BaseException:
                pass
            if not isinstance(error, ASRLoadError) and isinstance(error, (OSError, RuntimeError)):
                raise ASRLoadError(str(error), native_code=getattr(error, "native_code", None)) from error
            raise

    def _check_process(self):
        if os.getpid() != self._pid:
            raise ASRStateError("Inherited model; initialize after spawn/exec", native_code=-6)

    def _acquire(self):
        self._check_process()
        if not self._request.acquire(blocking=False):
            raise ASRStateError("Model is busy; wait for the active call", native_code=-6)
        if self._closing.is_set():
            self._request.release()
            raise ASRStateError("Model is closed", native_code=-6)

    def _options(self, language, hotwords, prompt, max_new_tokens):
        if language is not None and (not isinstance(language, str) or language != "auto"):
            raise ASRUnsupportedError('language supports None or "auto"')
        if hotwords is not None and (
            not isinstance(hotwords, (list, tuple))
            or any(not isinstance(word, str) for word in hotwords)
        ):
            raise TypeError("hotwords must be a list/tuple of strings or None")
        limit = generation_limit(max_new_tokens, default=128, maximum=2048)
        prompt_bytes = text(
            "Please transcribe this audio." if prompt is None else prompt,
            "prompt", 16384,
        )
        plan = make_hotword_processor(
            self._processor.tokenizer, hotwords, topk=50,
            start_boost=6.0, continuation_boost=8.0,
        )
        return prompt_bytes, limit, plan

    def _prepare_input(self, audio, sample_rate, channel, prompt):
        pcm, duration = prepare_audio(
            audio, sample_rate=sample_rate, channel=channel, max_duration_s=30
        )
        if not 201 <= pcm.size <= 480000:
            raise ValueError("Effective audio must contain 201..480000 model-rate samples")
        with self._processor._operation():
            return self._processor._prepare_pcm(pcm, prompt), duration

    def transcribe(
        self,
        audio,
        *,
        sample_rate=None,
        language=None,
        hotwords=None,
        channel=None,
        prompt=None,
        max_new_tokens=None,
    ):
        """Synchronously transcribe one complete file or PCM array."""
        self._acquire()
        try:
            prompt_bytes, limit, plan = self._options(language, hotwords, prompt, max_new_tokens)
            inputs, duration = self._prepare_input(audio, sample_rate, channel, prompt_bytes)
            output = self._model.generate(
                **inputs, max_new_tokens=limit, do_sample=False,
                logits_processor=[plan] if plan is not None else None,
            )
            from ._web import record_tokens

            generated = output[0, inputs["input_ids"].shape[1]:]
            result = self._result(generated, duration)
            record_tokens(generated)
            return result
        finally:
            self._request.release()

    def transcribe_batch(
        self,
        audios,
        *,
        sample_rate=None,
        language=None,
        hotwords=None,
        channel=None,
        prompt=None,
        max_new_tokens=None,
    ):
        """Transcribe 2–32 complete files or floating-point PCM arrays as one batch.

        Options apply to every input. The call waits for the entire batch and
        returns one TranscriptionResult per input, in input order.
        """
        self._acquire()
        try:
            if not isinstance(audios, (list, tuple)):
                raise TypeError("audios must be a list or tuple of complete audio inputs")
            items = tuple(audios)
            if not 2 <= len(items) <= 32:
                raise ValueError("audios must contain 2 to 32 complete audio inputs")
            prompt_bytes, limit, plan = self._options(language, hotwords, prompt, max_new_tokens)
            prepared = [
                self._prepare_input(audio, sample_rate, channel, prompt_bytes) for audio in items
            ]
            rows = [
                (inputs["input_ids"][0], inputs["input_features"][0])
                for inputs, _ in prepared
            ]
            output = self._model._generate_prepared_batch(
                rows, limit, plan
            )
            return [
                self._result(ids, duration)
                for ids, (_, duration) in zip(output, prepared, strict=True)
            ]
        finally:
            self._request.release()

    def _result(self, ids, duration):
        # The engine validates the native completion status against EOS and the limit.
        reason = "stop" if ids[-1] == self._processor.tokenizer.eos_token_id else "length"
        transcript = self._processor.decode(
            ids, skip_special_tokens=True, allow_incomplete_tail=reason == "length"
        )
        return TranscriptionResult(transcript, None, duration, reason)

    def close(self):
        self._check_process()
        self._closing.set()
        with self._request:
            if self._closed:
                return
            errors = []
            for owner in (self._model, self._processor):
                if owner is not None:
                    try:
                        owner.close()
                    except BaseException as error:
                        errors.append(error)
            if errors:
                raise errors[0]
            self._closed = True

    def __enter__(self):
        self._acquire()
        self._request.release()
        return self

    def __exit__(self, *_):
        self.close()
