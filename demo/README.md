# Web Demo

Complete [Install](../README.md#1-install) and [Get the model](../README.md#2-get-the-model). From the Demo root, run `bash demo/run.sh --port 8766` and open `http://<device-ip>:8766/`. Stop the server with `Ctrl+C`.

**Live transcription** starts on the home page.

- **Audio file.** Select **Open file** and choose a WAV, MP3, FLAC, Ogg Opus or M4A file supported by your browser. Listen using the audio controls. Select **Upload and transcribe** to send the entire audio from the beginning to the Demo server. The limit is 64 MiB and 10 minutes. If the browser cannot decode a file, select it again or choose a different format.
- **Video.** Select **Open file** to play a local video or **Use demo** to return to the built-in videos. **Start transcription** sends the sound in sync with playback. Pause confirms received speech. Seeking during transcription starts a new session from the new position.
- **Microphone.** Select **Microphone**, then **Start transcription** and allow browser access. A secure page (HTTPS or localhost) is required. **Stop** ends capture.

All three sources use the same transcript panel. Preview text can change before confirmation. Copy and export include confirmed text. Hotwords provide recognition hints.

**Batch transcription** lets you listen to the included 25-second sample independently of inference. Choose 2–32 inputs and select **Start all**. The same sample is submitted as independent inputs in one SDK request and all transcripts appear together.

- Per-input waiting RTF divides the browser time from **Start all** to receipt of the batch result by the sample duration. It includes HTTP, queueing and result retrieval and is shared by all inputs in the same batch.
- Batch throughput RTF divides the public SDK call time by the combined duration of all inputs. The SDK time includes audio preparation, inference and decoding. Model startup and sample playback are outside both measurements.

See [Media attribution](../docs/MEDIA_ATTRIBUTION.md) for sources and licenses. The service listens on the configured interface and has no built-in authentication.
