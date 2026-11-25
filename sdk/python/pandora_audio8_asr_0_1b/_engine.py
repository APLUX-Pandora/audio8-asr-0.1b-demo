"""Audio8 API aligned with its original processor/generate/decode flow."""
import ctypes as C
import json
import numpy as np

from ._audio import prepare_audio
from ._binding import Owner, GenerationRequest, NativeError, StreamOptions, TokenCallback, BatchRequest, ids, text
from ._hotword import _HotwordProcessor


class _Tokenizer:
    def __init__(self, owner, model_key):
        self._owner = owner
        self._model_key = model_key
        self.padding_side = "right"

    @property
    def pad_token_id(self):
        return 151643

    @property
    def eos_token_id(self):
        with self._owner._operation():
            return 151645

    @property
    def all_special_ids(self):
        p = self._owner
        with p._operation():
            return p._array(p._lib.a8_processor_special_ids, "int32").tolist()

    def encode(self, value, add_special_tokens=False):
        if add_special_tokens is not False:
            raise NotImplementedError("Only add_special_tokens=False is supported")
        p = self._owner
        value = text(value, "tokenizer input", 1048576)
        with p._operation():
            return p._array(p._lib.a8_processor_encode, "int32", value).tolist()

    def convert_ids_to_tokens(self, token_id):
        if type(token_id) is not int or not 0 <= token_id < 151936:
            raise ValueError("Expected one vocabulary token ID")
        p = self._owner
        with p._operation():
            return p._string(p._lib.a8_processor_symbol, token_id)

    def decode(self, token_ids, *, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        # Transformers streamers pass an accumulated Python token list.
        if isinstance(token_ids, (list, tuple)):
            if any(type(value) is not int for value in token_ids):
                raise TypeError("Token IDs must be integers")
            token_ids = np.asarray(token_ids, dtype=np.int64)
        return self._owner.decode(
            token_ids, skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces)

    def _compile_hotwords(self, words, k, start, continuation):
        p = self._owner
        payload = text(json.dumps(words, ensure_ascii=False), "hotword input", 1048576)
        with p._operation():
            return p._string(p._lib.a8_processor_hotword_plan, payload, k, start, continuation).encode()


class _Processor(Owner):
    @classmethod
    def _from_model(cls, model):
        obj = cls._load(model=model)
        try:
            obj.tokenizer = _Tokenizer(obj, model._hotword_key)
            return obj
        except BaseException:
            obj.close()
            raise

    def apply_chat_template(self, conversation, *, add_generation_prompt=True,
                            return_tensors="np", sampling_rate=16000,
                            audio_padding="longest", audio_max_length=None,
                            text_kwargs=None):
        options = dict(add_generation_prompt=add_generation_prompt,
                       return_tensors=return_tensors, sampling_rate=sampling_rate,
                       audio_padding=audio_padding, audio_max_length=audio_max_length,
                       text_kwargs=text_kwargs)
        with self._operation():
            if isinstance(conversation, list) and conversation and isinstance(conversation[0], list):
                if not 1 <= len(conversation) <= 32:
                    raise ValueError("Supported batch size is 1..32")
                if self.tokenizer.padding_side not in ("left", "right"):
                    raise ValueError("padding_side must be left or right")
                prepared = [self._apply_single(row, **options) for row in conversation]
                width = max(x["input_ids"].shape[1] for x in prepared)
                frames = max(x["input_features"].shape[2] for x in prepared)
                tokens = np.full((len(prepared), width), 151643, dtype=np.int64)
                mask = np.zeros_like(tokens)
                features = np.zeros((len(prepared), 128, frames), dtype=np.float32)
                lengths = []
                for i, row in enumerate(prepared):
                    n, t = row["input_ids"].shape[1], row["input_features"].shape[2]
                    start = width - n if self.tokenizer.padding_side == "left" else 0
                    tokens[i, start:start+n] = row["input_ids"][0]
                    mask[i, start:start+n] = 1
                    features[i, :, :t] = row["input_features"][0]
                    lengths.append(t)
                return dict(input_ids=tokens, attention_mask=mask,
                            input_features=features, audios=features,
                            feature_lens=np.asarray(lengths, dtype=np.int64))
            return self._apply_single(conversation, **options)

    def _apply_single(self, conversation, *, add_generation_prompt=True,
                            return_tensors="np", sampling_rate=16000,
                            audio_padding="longest", audio_max_length=None,
                            text_kwargs=None):
        self._check_process()
        if add_generation_prompt is not True or return_tensors != "np" or audio_padding != "longest":
            raise NotImplementedError("Use generation prompt, NumPy tensors and longest padding")
        if type(sampling_rate) is not int or sampling_rate != 16000:
            raise ValueError("sampling_rate is the processor target rate and must be 16000")
        if text_kwargs is not None:
            if not isinstance(text_kwargs, dict):
                raise TypeError("text_kwargs must be a dictionary or None")
            allowed = {"padding": "longest", "truncation": False, "add_special_tokens": False}
            if any(k not in allowed or type(v) is not type(allowed[k]) or v != allowed[k] for k, v in text_kwargs.items()):
                raise NotImplementedError("Text truncation/extra tokenizer options are unsupported")
        if not isinstance(conversation, list) or len(conversation) != 1:
            raise ValueError("Each item must contain one user message with one audio")
        message = conversation[0]
        if not isinstance(message, dict) or set(message) != {"role", "content"} or message["role"] != "user":
            raise ValueError("Expected a user message with role/content only")
        parts = message["content"]
        if not isinstance(parts, list) or len(parts) != 2 or not all(isinstance(x, dict) for x in parts):
            raise ValueError("Expected one audio part then one text part")
        audio, task = parts
        if task.get("type") != "text" or set(task) != {"type", "text"}:
            raise ValueError("Expected a text part")
        prompt = text(task["text"], "task text", 16384)
        if audio.get("type") != "audio" or set(audio) not in ({"type", "path"}, {"type", "array"}):
            raise ValueError("Expected one local path or array audio part")
        if audio_max_length is not None and (type(audio_max_length) is not int or not 201 <= audio_max_length <= 480000):
            raise ValueError("audio_max_length must be 201..480000 samples or None")
        source = audio.get("path") if "path" in audio else audio["array"]
        if audio_max_length is None:
            pcm, _ = prepare_audio(source, sample_rate=16000 if "array" in audio else None, max_duration_s=30)
        elif "path" in audio:
            # Decode only the explicitly requested prefix; resampling uses file metadata.
            import soundfile as sf
            with sf.SoundFile(source) as stream:
                count = int(np.ceil(audio_max_length * stream.samplerate / 16000))
                raw = stream.read(frames=count, dtype="float32", always_2d=True)
                pcm, _ = prepare_audio(raw, sample_rate=int(stream.samplerate))
            pcm = pcm[:audio_max_length]
        else:
            if not isinstance(source, np.ndarray):
                raise TypeError("Audio arrays must be NumPy floating PCM")
            pcm, _ = prepare_audio(source[:audio_max_length], sample_rate=16000)
        if not 201 <= pcm.size <= 480000:
            raise ValueError("Effective audio must contain 201..480000 model-rate samples")
        return self._prepare_pcm(pcm, prompt)

    def _prepare_pcm(self, pcm, prompt):
        # Input adaptation has already validated the source. Preserve finite FIR
        # overshoot when passing model-rate PCM into the native frontend.
        self._check(self._lib.a8_processor_prepare(self._slot, pcm.ctypes.data_as(C.POINTER(C.c_float)), pcm.size, prompt))
        tokens = self._array(self._lib.a8_processor_get_ids, "int32").astype(np.int64).reshape(1, -1)
        features = self._array(self._lib.a8_processor_get_features, "float32").reshape(1, 128, -1)
        return {"input_ids": tokens, "attention_mask": np.ones_like(tokens),
                "input_features": features, "audios": features}

    def batch_decode(self, sequences, *, skip_special_tokens=False,
                     clean_up_tokenization_spaces=False):
        values = ids(sequences, ndim=2)
        return [self.decode(row, skip_special_tokens=skip_special_tokens,
                            clean_up_tokenization_spaces=clean_up_tokenization_spaces)
                for row in values]

    def decode(self, token_ids, *, skip_special_tokens=False,
               clean_up_tokenization_spaces=False, allow_incomplete_tail=False):
        if type(skip_special_tokens) is not bool:
            raise TypeError("skip_special_tokens must be bool")
        if clean_up_tokenization_spaces is not False:
            raise NotImplementedError("Tokenization-space cleanup is not supported")
        values = ids(token_ids, ndim=1)
        with self._operation():
            return self._string(self._lib.a8_processor_decode,
                                values.ctypes.data_as(C.POINTER(C.c_int32)), values.size,
                                int(skip_special_tokens) | (2 if allow_incomplete_tail else 0))


class _GenerationModel(Owner):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *, runtime=None, library=None):
        key = object()
        model = cls._load(pretrained_model_name_or_path, runtime, library)
        model._hotword_key = key
        return model

    def _hotword_payload(self, plan):
        if plan is None:
            return None
        if not isinstance(plan, _HotwordProcessor):
            raise NotImplementedError("Unsupported logits processor")
        if plan.owner_key is not self._hotword_key:
            raise ValueError("Hotword plan belongs to a different model")
        return plan.payload

    def _generate_batch(self, tokens, mask, features, lengths, limit, plan):
        count, width = tokens.shape
        if mask is None:
            mask = np.ones_like(tokens)
        rows = []
        for i in range(count):
            valid = int(mask[i].sum())
            if not valid or not np.all(mask[i, :width-valid] == 0) or not np.all(mask[i, width-valid:] == 1):
                raise ValueError("Batch generation requires contiguous left padding")
            if not np.all(tokens[i, :width-valid] == 151643):
                raise ValueError("Masked input IDs must contain the upstream PAD token")
            rows.append((tokens[i, width-valid:], features[i, :, :int(lengths[i])]))
        generated = self._generate_prepared_batch(rows, limit, plan)
        output = np.full((count, width + max(len(row) for row in generated)), 151643, dtype=np.int64)
        output[:, :width] = tokens
        for i, row in enumerate(generated):
            output[i, width:width+len(row)] = row
        return output

    def _generate_prepared_batch(self, rows, limit, plan):
        """Keep each prepared row independent and borrow its buffers through one native batch."""
        payload = self._hotword_payload(plan)
        buffers = [
            (np.ascontiguousarray(tokens, dtype=np.int32), np.ascontiguousarray(features))
            for tokens, features in rows
        ]
        count = len(buffers)
        requests = (GenerationRequest * count)(*[
            GenerationRequest(
                C.sizeof(GenerationRequest), 1, tokens.ctypes.data_as(C.POINTER(C.c_int32)),
                tokens.size, None, features.ctypes.data_as(C.POINTER(C.c_float)),
                features.shape[1], limit, payload,
            )
            for tokens, features in buffers
        ])
        request = BatchRequest(C.sizeof(BatchRequest), 2, requests, count)
        result = C.c_void_p()
        with self._operation():
            try:
                self._check(self._lib.a8_generate_batch(self._slot, C.byref(request), C.byref(result)))
                generated = []
                for i in range(count):
                    n, reason = C.c_size_t(), C.c_uint32()
                    get = self._lib.a8_batch_result_get
                    self._check(get(result, i, None, 0, C.byref(n), C.byref(reason)))
                    if not 1 <= n.value <= limit:
                        raise NativeError(-5, "Invalid batch result length")
                    output = np.empty(n.value, dtype=np.int32)
                    self._check(get(result, i, output.ctypes.data_as(C.POINTER(C.c_int32)),
                                    output.size, C.byref(n), C.byref(reason)))
                    if (n.value != output.size or reason.value not in (1, 2)
                            or (reason.value == 1 and output[-1] != 151645)
                            or (reason.value == 2 and (output[-1] == 151645 or output.size != limit))):
                        raise NativeError(-5, "Invalid batch completion status")
                    generated.append(output)
                return generated
            except BaseException as error:
                if not isinstance(error, ValueError):
                    self._failed = True
                raise
            finally:
                if result.value:
                    self._lib.a8_batch_result_destroy_owned(C.byref(result))

    def generate(self, input_ids, *, attention_mask=None, input_features=None, audios=None,
                 max_new_tokens=128, do_sample=False, logits_processor=None, streamer=None,
                 feature_lens=None):
        self._check_process()
        if do_sample is not False:
            raise NotImplementedError("Only greedy do_sample=False is supported")
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 2048:
            raise ValueError("max_new_tokens must be an integer in 1..2048")
        if streamer is not None and (not callable(getattr(streamer, "put", None)) or
                                     not callable(getattr(streamer, "end", None))):
            raise TypeError("streamer must provide callable put(token_ids) and end()")
        tokens = ids(input_ids, ndim=2)
        if tokens.shape[1] + max_new_tokens > 2048:
            raise ValueError("Input and requested output must fit CL2048")
        mask = None
        if attention_mask is not None:
            mask = ids(attention_mask, ndim=2)
            if mask.shape != tokens.shape or not np.isin(mask, [0, 1]).all():
                raise ValueError("attention_mask must match input_ids and contain 0/1")
        if input_features is not None and audios is not None:
            if not isinstance(audios, np.ndarray) or not isinstance(input_features, np.ndarray) or not np.array_equal(input_features, audios):
                raise ValueError("Feature aliases disagree")
        features = input_features if input_features is not None else audios
        if not isinstance(features, np.ndarray) or features.dtype != np.dtype("float32"):
            raise TypeError("input_features/audios must be float32 NumPy features")
        if features.ndim != 3 or features.shape[:2] != (tokens.shape[0], 128) or not 1 <= features.shape[2] <= 3000 or not np.isfinite(features).all():
            raise ValueError("Expected finite [B,128,T] audio features")
        features = np.ascontiguousarray(features)
        plan = None
        if logits_processor is not None:
            if not isinstance(logits_processor, (list, tuple)) or len(logits_processor) > 1:
                raise NotImplementedError("Use zero or one native hotword processor; arbitrary callbacks are unsupported")
            if logits_processor:
                plan = logits_processor[0]
                if plan is None:
                    raise NotImplementedError("Unsupported logits processor")
        if feature_lens is None:
            lengths = np.full(tokens.shape[0], features.shape[2], dtype=np.int64)
        else:
            if (not isinstance(feature_lens, np.ndarray) or
                feature_lens.dtype not in (np.dtype("int32"), np.dtype("int64")) or
                feature_lens.shape != (tokens.shape[0],) or
                np.any(feature_lens < 1) or np.any(feature_lens > features.shape[2])):
                raise ValueError("feature_lens must be integer [B] real mel frame lengths")
            lengths = feature_lens
        if tokens.shape[0] > 1:
            if streamer is not None:
                raise NotImplementedError("Batch generation is synchronous")
            return self._generate_batch(tokens, mask, features, lengths, max_new_tokens, plan)
        if mask is not None and not np.all(mask == 1):
            raise ValueError("A single-item request must be unpadded")
        features = np.ascontiguousarray(features[:, :, :int(lengths[0])])
        request = GenerationRequest(C.sizeof(GenerationRequest), 1,
                    tokens.ctypes.data_as(C.POINTER(C.c_int32)), tokens.size,
                    None if mask is None else mask.ctypes.data_as(C.POINTER(C.c_int32)),
                    features.ctypes.data_as(C.POINTER(C.c_float)), features.shape[2], max_new_tokens,
                    self._hotword_payload(plan))
        callback_errors = []
        stream_options = None
        if streamer is not None:
            try:
                stream_fn = self._lib.a8_generate_prepared_stream
            except AttributeError as error:
                raise RuntimeError("Streaming requires the paired native streaming candidate") from error
            stream_fn.argtypes = [C.c_void_p, C.POINTER(GenerationRequest), C.POINTER(StreamOptions)]
            stream_fn.restype = C.c_int

            @TokenCallback
            def on_token(token_id, _):
                try:
                    streamer.put(np.array([token_id], dtype=np.int64))
                    return 0
                except BaseException as error:
                    callback_errors.append(error)
                    return 1

            stream_options = StreamOptions(C.sizeof(StreamOptions), 1, on_token, None)
        with self._operation():
            streaming_started = False
            execution_started = False
            primary_error = None
            try:
                if streamer is not None:
                    streaming_started = True
                    streamer.put(tokens.astype(np.int64).copy())
                execution_started = True
                if stream_options is None:
                    code = self._lib.a8_generate_prepared(self._slot, C.byref(request))
                else:
                    code = stream_fn(self._slot, C.byref(request), C.byref(stream_options))
                if callback_errors:
                    raise callback_errors[0]
                self._check(code)
                output = self._array(self._lib.a8_get_tokens, "int32").astype(np.int64)
                reason, count = C.c_uint32(), C.c_size_t()
                self._check(self._lib.a8_get_generation_status(self._slot, C.byref(reason), C.byref(count)))
                if count.value != len(output) or not len(output) or len(output) > max_new_tokens:
                    raise NativeError(-5, "Invalid native generation count")
                if (reason.value == 1 and output[-1] != 151645) or (reason.value == 2 and (output[-1] == 151645 or len(output) != max_new_tokens)) or reason.value not in (1, 2):
                    raise NativeError(-5, "Invalid native generation completion")
                return np.concatenate((tokens.astype(np.int64).reshape(-1), output)).reshape(1, -1)
            except BaseException as error:
                primary_error = error
                # Native input validation is recoverable; callback failures after
                # execution and every other execution error require recreation.
                if execution_started and (callback_errors or not isinstance(error, ValueError)):
                    self._failed = True
                raise
            finally:
                if streaming_started:
                    if primary_error is not None and callable(getattr(streamer, "fail", None)):
                        try:
                            streamer.fail(primary_error)
                        except BaseException:
                            pass
                    else:
                        try:
                            streamer.end()
                        except BaseException as end_error:
                            if primary_error is None:
                                if callable(getattr(streamer, "fail", None)):
                                    try:
                                        streamer.fail(end_error)
                                    except BaseException:
                                        pass
                                raise
