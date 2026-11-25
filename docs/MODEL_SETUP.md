# Model setup

## Device requirements

- A compatible Linux device with Aidlux LMS (`sudo lms check` returns true), drivers, DSP firmware and BSP/FastRPC support.
- Python 3.10–3.12 in the Demo environment (see [Install](../README.md#1-install)).
- The SDK includes its QNN runtime libraries. Do not install QNN separately or set QNN paths.

## Targets

Check the device ID with `cat /sys/devices/soc0/soc_id` and pick its target folder:

| SoC | SoC ID | Target folder |
| --- | --- | --- |
| QCS8550 | 603 | `qcs8550_qnn248` |
| QCS9075 | 676 | `qcs9075_qnn248` |
| QCS8625 | 702 | `qcs8625_qnn248` |
| QCS8275 | 675 | `qcs8275_qnn248` |
| QCS6490 | 498 | `qcs6490_qnn248` |

## Download

In the second command, replace `<target-folder>` with the folder for your SoC.

From the Demo root, in the activated environment, download the shared resources once and then one target:

```bash
hf download APLUX-Pandora/Audio8-ASR-0.1B --local-dir models --exclude "targets/*" --exclude "media/*" --exclude README.md --exclude .gitattributes
hf download APLUX-Pandora/Audio8-ASR-0.1B --local-dir models --include "targets/<target-folder>/*"
```

Run `hf auth login` first if the repository is private. Keep one target per model folder: to switch devices, move the old `targets/` folder out or download into a new folder.

For QCS8550, use this as the second command:

```bash
hf download APLUX-Pandora/Audio8-ASR-0.1B --local-dir models --include "targets/qcs8550_qnn248/*"
```

## Folder layout

After the download, `models/` looks like this:

```text
models/
├── models.json
├── SHA256SUMS
├── tokenizer.json
├── embedding.bin
├── prefix.bin
├── position_cos.bin
├── position_sin.bin
├── frontend/
│   ├── hann.bin
│   └── mel_filters.bin
└── targets/
    └── qcs8550_qnn248/
        ├── encoder.bin.aidem
        ├── projector.bin.aidem
        └── lm.bin.aidem
```

The SDK reads `models.json`, checks that the shared resources and one complete target are present, and reports what is missing. `hf download` verifies each large file's checksum.

At load time, the SDK uses the detected SoC to pick the target, checks file sizes, and decrypts the model. QCS8275 can also use the QCS8625 target when its own target is unavailable.

The QCS6490 target supplies its own `prefix.bin`. The other targets use the shared prefix.

## Troubleshooting

| Message or symptom | What to do |
| --- | --- |
| The SDK reports a missing resource or target | Run both download commands for the same model directory and select the target for your SoC. |
