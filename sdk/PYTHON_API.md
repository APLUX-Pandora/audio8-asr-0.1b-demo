# Python API

`pandora_audio8_asr_0_1b` · Load Audio8-ASR-0.1B and transcribe audio files or floating-point PCM.

## Quick start

```python
from pandora_audio8_asr_0_1b import ASRModel

with ASRModel("models") as model:
    result = model.transcribe("examples/data/lecture_10s.wav")
    print(result.text, result.finish_reason)
```

## API overview

| Name | Signature | Returns | Purpose |
| --- | --- | --- | --- |
| `ASRModel` | `(model_dir, *, runtime=None)` | `ASRModel` | Load a local model folder |
| `ASRModel.from_pretrained` | `(model_dir, *, runtime=None)` | `ASRModel` | Load a local model folder |
| `transcribe` | `(audio, *, sample_rate=None, language=None, hotwords=None, channel=None, prompt=None, max_new_tokens=None)` | `TranscriptionResult` | Recognize one clip |
| `transcribe_batch` | `(audios, *, sample_rate=None, language=None, hotwords=None, channel=None, prompt=None, max_new_tokens=None)` | `list[TranscriptionResult]` | Recognize a batch of complete clips |
| `close` | `()` | `None` | Release the model |

## Parameters

### `transcribe`

| Parameter | Type | Default | Range / unit | Meaning |
| --- | --- | --- | --- | --- |
| `audio` | path or NumPy float32/float64 array | required | up to 30 seconds | Local audio file or PCM shaped `(samples,)` or `(samples, channels)` |
| `sample_rate` | int or None | file rate | Hz | Original PCM sample rate, required for arrays. Files supply their own rate |
| `language` | str or None | None | None or `"auto"` | Both select native multilingual transcription. Explicit language codes raise an error |
| `hotwords` | list or tuple of strings | None | model limits | Hints for this request. None or an empty list disables hints |
| `channel` | int or None | None | zero-based | Use mono as-is or average stereo when omitted. Choose a channel for audio with more than two channels |
| `prompt` | str or None | None | optional | Override the model's transcription prompt |
| `max_new_tokens` | int or None | None | 1–2048 | Limit generation. None uses the model's default of 128 tokens |

### `transcribe_batch`

Supply a list or tuple of 2–32 complete audio inputs. Each item follows the same file or floating PCM rules and 30-second limit as `transcribe`. Keyword options apply to every item. With `sample_rate=None`, each file uses its own rate. If you supply a rate, it applies to every PCM item and must agree with any file's metadata.

The SDK processes the inputs in one synchronous batch and returns one `TranscriptionResult` per input, in input order. Invalid inputs fail the entire request. The two inference calls accept complete clips and reject streaming arguments such as `stream` or `streamer`.

```python
from pandora_audio8_asr_0_1b import ASRModel

files = ("examples/data/lecture_10s.wav", "examples/data/lecture_tail4s.wav")
with ASRModel.from_pretrained("models") as model:
    results = model.transcribe_batch(files)

for audio, result in zip(files, results, strict=True):
    print(audio, result.text, result.finish_reason)
```

## Results and cleanup

### `TranscriptionResult`

| Field | Type | Meaning |
| --- | --- | --- |
| `text` | str | Decoded transcription |
| `duration_s` | float | Original input duration in seconds |
| `language` | str or None | None for this model |
| `finish_reason` | str | `stop` at EOS or `length` when the token limit is reached |
| `is_complete` | bool | True when the finish reason is `stop` |

### Errors

| Exception | When |
| --- | --- |
| `TypeError`, `ValueError` | Invalid audio or parameter |
| `ASRUnsupportedError` | Unsupported language selection |
| `ASRLoadError` | Model, library or Runtime initialization fails |
| `ASRInferenceError` | Native failure during inference, including NPU (`native_code=-3`), I/O (`-4`) or internal errors (`-5`) |
| `ASRStateError` | Model is closed, busy, inherited across a fork or requires recreation after an execution failure (`native_code=-6`) |

When a native operation fails during model initialization, `ASRLoadError` carries its original `native_code` if available.

### Lifecycle

Use one model for successive calls. A concurrent inference request raises `ASRStateError`. Returned results remain readable after closing the model. Each call returns independent text and starts with its own hotwords. Use `with` or call `close()` to wait for an active call and release the model. Closing twice is safe. A supplied Runtime is owned by the caller.

## Advanced

The model returns text from multilingual speech. Recognition accuracy depends on the input, and hints do not guarantee a match. The SDK reports no detected language, timestamps or confidence scores. For live captions, the application can pace input, form short windows and call `transcribe` per window. See the [examples](../examples/README.md) for complete-clip and application-level realtime usage.
