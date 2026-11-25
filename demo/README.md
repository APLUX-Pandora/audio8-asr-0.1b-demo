# Web Demo

Complete [Install](../README.md#1-install) and [Get the model](../README.md#2-get-the-model). From the Demo root, run `bash demo/run.sh --port 8766` and open `http://<device-ip>:8766/`. Stop the server with `Ctrl+C`.

**Live transcription** starts on the home page. Choose a built-in video or a local audio or video file. The Demo collects audio during playback, uses voice activity to choose short windows, and transcribes each window synchronously. Preview text can change before confirmation. The subtitle times refer to playback windows. The model can misrecognize speech, and hotwords only provide hints.

**Batch transcription** lets you listen to the included 25-second sample independently of inference. Choose 2–32 inputs and select **Start all**. The same sample is submitted as independent inputs in one SDK request and all transcripts appear together.

- Per-input waiting RTF divides the browser time from **Start all** to receipt of the batch result by the sample duration. It includes HTTP, queueing and result retrieval and is shared by all inputs in the same batch.
- Batch throughput RTF divides the public SDK call time by the combined duration of all inputs. The SDK time includes audio preparation, inference and decoding. Model startup and sample playback are outside both measurements.

See [Media attribution](../docs/MEDIA_ATTRIBUTION.md) for sources and licenses. The service listens on the configured interface and has no built-in authentication.
