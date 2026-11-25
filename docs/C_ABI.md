# C ABI

`sdk/include/audio8.h` · `sdk/lib/libaudio8.so` · Prepare audio and run offline recognition.

## Build and link

In the same Python environment, install the Runtime and SDK from the Demo root before compiling:

```bash
python -m pip install sdk/wheels/pandora_runtime-*.whl
python -m pip install ./sdk
```

From the Demo root, compile against the shipped header and library; the library carries its QNN runtime:

```bash
runtime_dir=$(python -c "import pathlib,pandora_runtime; print(pathlib.Path(pandora_runtime.__file__).parent)")
cc -std=c99 -I sdk/include your_app.c -L sdk/lib -L"$runtime_dir" \
  -Wl,--no-as-needed -l:libpandora_runtime.so.5 -Wl,--as-needed -Wl,-rpath,"$runtime_dir" \
  -laudio8 -Wl,-rpath,'$ORIGIN/sdk/lib' -o your_app
```

For deployments without Python, place the wheel's `libpandora_runtime.so.5` where the system loader can find it.

## Lifecycle

```text
Runtime → processor + recognizer → prepare → generate → read → destroy both → destroy Runtime
```

Reuse the processor and recognizer for another clip. For the simple PCM entry points, reset before feeding the next clip. On failure, release the owned handles before recreating them.

## Functions

The Runtime wheel provides the `pandora_runtime_*` functions.

| Function | Purpose | Returns |
| --- | --- | --- |
| `a8_prepare_process` | Prepare the process for loading a protected model | Status code |
| `a8_get_protection_stats` | Read protected-buffer counters | Status code |
| `a8_create` | Create a recognizer with default Runtime settings | Status code |
| `a8_destroy` | Destroy a recognizer | void |
| `a8_create_with_runtime` | Attach a recognizer to a Runtime | Status code |
| `a8_destroy_owned` | Destroy and clear an owned recognizer slot | Status code |
| `a8_set_hotwords` | Set or clear recognition hints | Status code |
| `a8_reset` | Reset accumulated input and generation | Status code |
| `a8_feed` | Append mono 16 kHz floating PCM to one offline request | Status code |
| `a8_finish` | Run the offline request over accumulated PCM | Status code |
| `a8_get_tokens` | Copy generated token IDs | Status code |
| `a8_get_text` | Copy the decoded result | Status code |
| `a8_token_bytes` | Copy one token's bytes | Status code |
| `a8_processor_create` | Create a processor attached to a Runtime | Status code |
| `a8_processor_create_from_model` | Create a processor sharing the recognizer’s loaded tokenizer, audio frontend and Runtime | Status code |
| `a8_processor_destroy_owned` | Destroy and clear an owned processor slot | Status code |
| `a8_processor_prepare` | Build features and prompt tokens from PCM | Status code |
| `a8_processor_get_ids` | Copy prepared token IDs | Status code |
| `a8_processor_get_features` | Copy prepared features | Status code |
| `a8_processor_encode` | Encode text as token IDs | Status code |
| `a8_processor_special_ids` | Copy special token IDs | Status code |
| `a8_processor_symbol` | Copy one token's vocabulary symbol | Status code |
| `a8_processor_decode` | Decode token IDs as UTF-8 | Status code |
| `a8_processor_hotword_plan` | Build a token-level hint plan | Status code |
| `a8_generate_prepared` | Generate from prepared audio features and tokens | Status code |
| `a8_generate_batch` | Generate for prepared input rows | Status code |
| `a8_batch_result_get` | Copy one row's tokens and completion reason | Status code |
| `a8_batch_result_destroy_owned` | Destroy and clear a batch result slot | Status code |
| `a8_get_generation_status` | Read the completion reason and generated token count | Status code |
| `a8_last_error` | Read the current thread's Audio8 error message | Borrowed string |
| `pandora_runtime_last_error` | Read the current thread's Runtime error message | Borrowed string |
| `pandora_runtime_create` | Create a Runtime from settings | Status code |
| `pandora_runtime_get_status` | Read Runtime status | Status code |
| `pandora_runtime_destroy` | Destroy and clear the Runtime slot | Status code |

**Batch requests**

Prepare 2–32 independent clips with the processor, copying each clip's IDs and features into a separate `a8_generation_request`. Use unpadded rows and keep those buffers alive through the synchronous call. Initialize `a8_batch_request` with its size, `A8_BATCH_ABI_VERSION`, the row array and its count. Leave reserved fields at zero.

Pass the address of a result pointer initialized to NULL to `a8_generate_batch`. The call validates all rows before inference and returns one owned result on success. Read each row with `a8_batch_result_get` using its zero-based input index. The result survives recognizer destruction. Release it with `a8_batch_result_destroy_owned`.

**Migrating from batch ABI 1**

Recompile against the current header and use the ABI 2 batch request with no callbacks. The prepared row layout and result functions are unchanged. The library retains the ABI 1 batch layout and legacy callback symbols for binaries built against the older interface. [ABI 1 compatibility declarations](../sdk/include/compat/audio8_abi1_compat.h) are provided for maintaining those integrations.

## Error codes

`a8_*` functions that return a status code use the Audio8 definitions.

| Code | Meaning | What the caller should do |
| --- | --- | --- |
| `A8_OK` (`0`) | Success | Continue |
| `A8_ERR_ARG` (`-1`) | Invalid input or insufficient output capacity | Check the arguments or query the required output size |
| `A8_ERR_MODEL` (`-2`) | Model or package error | Check the model files and device activation |
| `A8_ERR_NPU` (`-3`) | NPU or backend failure | Read the error and recreate the affected objects |
| `A8_ERR_IO` (`-4`) | I/O error | Check the input and file access |
| `A8_ERR_INTERNAL` (`-5`) | Internal failure | Read the error and release affected objects |
| `A8_ERR_STATE` (`-6`) | Invalid object state | Check the call order and ownership |

`pandora_runtime_*` functions that return a status code use the Pandora Runtime definitions.

| Code | Meaning | What the caller should do |
| --- | --- | --- |
| `PANDORA_RUNTIME_OK` (`0`) | Success | Continue |
| `PANDORA_RUNTIME_ERROR_ARGUMENT` (`-1`) | Invalid argument or ABI version | Check the argument values, structure size, and ABI version |
| `PANDORA_RUNTIME_ERROR_UNSUPPORTED` (`-2`) | Valid setting unsupported by this SDK or target | Choose a supported setting, such as an available NPU device |
| `PANDORA_RUNTIME_ERROR_STATE` (`-3`) | Wrong process (after fork), closed, or busy | Use an open Runtime created in the current process and serialize calls on it |
| `PANDORA_RUNTIME_ERROR_BACKEND` (`-4`) | QNN rejected a setting | Read the Runtime error message for details |
| `PANDORA_RUNTIME_ERROR_INTERNAL` (`-5`) | Internal failure | Read the Runtime error message for details |

## Threads and ownership

Serialize calls and destruction on each handle. A recognizer or processor retains its Runtime settings until destruction. It remains usable if the caller destroys the Runtime handle first. Input buffers must remain valid until the call returns. Output buffer storage belongs to the caller. Use a null destination and capacity zero for size queries where the header specifies it. Error strings are thread-local and borrowed. Fetch the error on the thread where the call failed.

A processor created from a recognizer keeps its own preparation results and can outlive the recognizer. Initialize the processor output slot to NULL before creation. Destroy the processor with `a8_processor_destroy_owned`.

Completion reason `1` means EOS and `2` means the token limit was reached. Use `A8_DECODE_ALLOW_INCOMPLETE_TAIL` only for length-limited output.

## Example

[The C example](../examples/c/generate.c) prepares one clip, reads its text and completion reason, and frees all owned resources.
