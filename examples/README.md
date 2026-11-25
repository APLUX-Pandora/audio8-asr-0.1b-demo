# Examples

## Overview

| Example | What it shows |
| --- | --- |
| [transcribe_file.py](python/transcribe_file.py) | Transcribe a WAV file |
| [transcribe_pcm.py](python/transcribe_pcm.py) | Pass floating PCM and its source sample rate |
| [transcribe_hotwords.py](python/transcribe_hotwords.py) | Apply hints to one request and reuse the model |
| [transcribe_batch.py](python/transcribe_batch.py) | Transcribe two complete clips in one batch and read results in input order |
| [transcribe_realtime.py](python/transcribe_realtime.py) | Read audio at playback pace and print one result per window |
| [generate.c](c/generate.c) | Prepare and transcribe one clip using the C ABI |

## Before you run

Complete [Install](../README.md#1-install) and [Get the model](../README.md#2-get-the-model). Run every command from the Demo root in the activated environment. Stop the Web Demo first; examples and the Web Demo share the NPU.

The examples use the [included lecture excerpt](data/README.md).

## Python examples

### transcribe_file.py

```bash
python examples/python/transcribe_file.py
```

Output

```text
Beautiful. Let's talk about developer experience for users of your. stop
```

### transcribe_pcm.py

```bash
python examples/python/transcribe_pcm.py
```

Output

```text
Beautiful. Let's talk about developer experience for users of your. stop
```

### transcribe_hotwords.py

```bash
python examples/python/transcribe_hotwords.py
```

Output:

```text
Default: Beautiful. Let's talk about developer experience for users of your.
Hotwords: ['developer experience', 'open']
With hints: Beautiful. Let's talk about developer experience for users of your open.
Default after hints: Beautiful. Let's talk about developer experience for users of your.
```

### transcribe_batch.py

```bash
python examples/python/transcribe_batch.py
```

Output

```text
examples/data/lecture_10s.wav: Beautiful. Let's talk about developer experience for users of your. (stop)
examples/data/lecture_tail4s.wav: Let's talk about developer experience for users of your open. (stop)
```

### transcribe_realtime.py

```bash
python examples/python/transcribe_realtime.py
```

The example reads the bundled ten-second WAV in 250 ms chunks at playback
pace. It prints the first eight-second window as soon as recognition finishes
and transcribes the final two seconds when the input ends. Pass a local mono
16 kHz PCM16 WAV path to use a different clip. A tail shorter than 201
samples is zero-padded to the SDK minimum.

Output

```text
[  0.00–  8.00 s] Beautiful. Let's talk about developer experience.
[  8.00– 10.00 s] For users of your open.
```

The example owns pacing and windowing. It calls the installed SDK's
synchronous `ASRModel.transcribe` once per window. These fixed windows have no
overlap or voice activity detection, so a word split at a boundary may be
incomplete. The Web Demo uses voice activity detection and can revise preview
text as playback advances.

## C example

### generate.c

Before building, make sure a C compiler is available and install the libsndfile development package for the audio headers and linker files.

```bash
sudo apt install libsndfile1-dev
```

Build against the headers and prebuilt library in `sdk/`:

```bash
runtime_dir=$(python -c "import pathlib,pandora_runtime; print(pathlib.Path(pandora_runtime.__file__).parent)")
cc -std=c99 -Wall -Wextra -Werror -I sdk/include examples/c/generate.c -L sdk/lib -L"$runtime_dir" \
  -Wl,--no-as-needed -l:libpandora_runtime.so.5 -Wl,--as-needed -Wl,-rpath,"$runtime_dir" \
  -laudio8 -lsndfile -Wl,-rpath,'$ORIGIN/../../sdk/lib' -o examples/c/generate
examples/c/generate
```

Output

```text
Beautiful. Let's talk about developer experience for users of your.
finish_reason=stop
```

## Reference

See the [Python API](../sdk/PYTHON_API.md) or [C ABI](../docs/C_ABI.md) for parameters, ownership and error handling.
