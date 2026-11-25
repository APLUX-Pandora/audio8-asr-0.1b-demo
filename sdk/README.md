# Audio8-ASR-0.1B SDK

Load Audio8-ASR-0.1B and synchronously transcribe complete audio files or floating-point PCM, one clip at a time or in batches of 2–32.

## What's included

| Path | Content |
| --- | --- |
| `pandora_audio8_asr_0_1b/` | Python package (`from pandora_audio8_asr_0_1b import ASRModel`) |
| `include/` | C header |
| `lib/` | Prebuilt native library with its QNN runtime |
| `SHA256SUMS` | Checksums of the shipped files |
| `wheels/` | Local Pandora Runtime wheel |

## Install

In your application's activated Python 3.10–3.12 environment, from the Demo root:

```bash
python -m pip install sdk/wheels/pandora_runtime-*.whl
python -m pip install ./sdk
```

The device must pass `sudo lms check`. The SDK does not install LMS or drivers and does not download models; see [Get the model](../README.md#2-get-the-model).

## Usage

```python
from pandora_audio8_asr_0_1b import ASRModel

with ASRModel("models") as model:
    result = model.transcribe("examples/data/lecture_10s.wav")
    print(result.text, result.finish_reason)
```

## Documentation

- [Python API](PYTHON_API.md)
- [Runtime](RUNTIME_API.md)
- [C ABI](../docs/C_ABI.md)
- [Examples](../examples/README.md)
- [Licensing](../LICENSING.md)
