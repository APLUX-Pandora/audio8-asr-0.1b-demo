# Built-in media

The Web Demo includes English, Japanese and Korean video excerpts and a 25-second speech sample for batch transcription. See [Media attribution](../../docs/MEDIA_ATTRIBUTION.md) for sources, edits and licenses.

The Batch transcription page uses the checked-in 16 kHz mono PCM WAV directly. To derive the same time interval from a local copy of the ten-minute FOSDEM excerpt, run `python3 demo/media/prepare_batch_audio.py --output-dir /path/to/output` from the Demo root with FFmpeg installed.
