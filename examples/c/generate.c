/* Standalone C99 inference using the delivered public ABI and libsndfile. */
#include "audio8.h"
#include <sndfile.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    a8_context *model = NULL;
    SF_INFO info = {0};
    SNDFILE *file = NULL;
    float *pcm = NULL;
    char *decoded = NULL;
    size_t output_count = 0, text_bytes = 0;
    const int max_new_tokens = 128;
    uint32_t reason = 0;
    int result = 1;
    const char *model_dir = argc == 3 ? argv[1] : "models";
    const char *audio_path = argc == 3 ? argv[2] : "examples/data/lecture_10s.wav";
    if (argc != 1 && argc != 3) {
        fprintf(stderr, "usage: generate MODEL_DIR MONO_16K_AUDIO\n");
        return 2;
    }
    file = sf_open(audio_path, SFM_READ, &info);
    if (!file) {
        fprintf(stderr, "Audio open failed: %s\n", sf_strerror(NULL));
        goto cleanup;
    }
    if (info.samplerate != 16000 || info.channels != 1 ||
        info.frames < 201 || info.frames > 480000) {
        fprintf(stderr, "Expected mono 16 kHz audio with 201..480000 samples\n");
        goto cleanup;
    }
    pcm = malloc((size_t)info.frames * sizeof(*pcm));
    if (!pcm) goto allocation_error;
    if (sf_readf_float(file, pcm, info.frames) != info.frames) {
        fprintf(stderr, "Incomplete audio read\n");
        goto cleanup;
    }
    if (sf_close(file)) {
        file = NULL;
        fprintf(stderr, "Audio close failed\n");
        goto cleanup;
    }
    file = NULL;
    if (a8_create(model_dir, &model) ||
        a8_feed(model, pcm, (size_t)info.frames) ||
        a8_finish(model, "Please transcribe this audio.", max_new_tokens) ||
        a8_get_generation_status(model, &reason, &output_count)) goto sdk_error;
    if (!output_count || output_count > (size_t)max_new_tokens ||
        (reason != 1 && reason != 2)) {
        fprintf(stderr, "Invalid generation completion\n");
        goto cleanup;
    }
    if (a8_get_text(model, NULL, 0, &text_bytes)) goto sdk_error;
    if (!text_bytes) {
        fprintf(stderr, "Missing decoded terminator\n");
        goto cleanup;
    }
    decoded = malloc(text_bytes);
    if (!decoded) goto allocation_error;
    if (a8_get_text(model, decoded, text_bytes, &text_bytes)) goto sdk_error;
    printf("%s\nfinish_reason=%s\n", decoded, reason == 1 ? "stop" : "length");
    result = 0;
    goto cleanup;
allocation_error:
    fprintf(stderr, "Allocation failed\n");
    goto cleanup;
sdk_error:
    fprintf(stderr, "SDK error: %s\n", a8_last_error());
cleanup:
    if (file && sf_close(file)) result = 1;
    if (a8_destroy_owned(&model)) result = 1;
    free(decoded);
    free(pcm);
    return result;
}
