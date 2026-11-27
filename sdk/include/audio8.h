#pragma once
#define A8_VERSION "1.1.0"
#include <stddef.h>
#include <stdint.h>
#include "pandora_runtime.h"
#if defined(__GNUC__)
#define A8_API __attribute__((visibility("default")))
#else
#define A8_API
#endif
#ifdef __cplusplus
extern "C" {
#endif
typedef struct a8_context a8_context;
enum { A8_OK=0,A8_ERR_ARG=-1,A8_ERR_MODEL=-2,A8_ERR_NPU=-3,A8_ERR_IO=-4,A8_ERR_INTERNAL=-5,A8_ERR_STATE=-6 };
/* Compatibility no-op: host debugging, limits and signals remain unchanged. */
A8_API int a8_prepare_process(void);
/* Process-local SDK-owned context-buffer diagnostics; never exposes addresses or model bytes.
 * Initialize struct_size=sizeof(a8_protection_stats), abi_version=1. Counters apply to
 * this process only. lock_failures is cumulative; last_lock_errno is the last failure.
 * Third-party runtime copies are outside this allocation scope. */
typedef struct {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t active_buffers;
    uint64_t dump_excluded_bytes;
    uint64_t locked_bytes;
    uint64_t lock_failures;
    int32_t last_lock_errno;
    uint32_t reserved;
} a8_protection_stats;
A8_API int a8_get_protection_stats(a8_protection_stats *out);


/* Synchronous, single caller per context. PCM is mono float32 16 kHz, <=30s. */
/* model_dir: nonempty NUL-terminated path, at most 4096 bytes. */
A8_API int a8_create(const char* model_dir,a8_context** result);
A8_API void a8_destroy(a8_context* c);
/* New owned slots must start NULL; close clears the slot before destruction. */
A8_API int a8_create_with_runtime(const char*, pandora_runtime_t*, a8_context**);
A8_API int a8_destroy_owned(a8_context**);
/* Decode-time hotwords; disabled by default. Settings survive reset; matching history does not.
 * UTF-8 words: <=128 items, <=256 bytes / 128 tokens each. count=0 disables (words may be NULL).
 * top_k=50 gates boosts to the original top candidates (0 allows all); boosts default to 6/8,
 * finite range 0..30 in dequantized logit units. Set between utterances, before feed/finish.
 * Invalid settings leave the previous configuration unchanged. No model/prefix changes. */
A8_API int a8_set_hotwords(a8_context* c,const char*const* words,size_t count,int top_k,float start_boost,float continuation_boost);
A8_API int a8_reset(a8_context* c);
A8_API int a8_feed(a8_context* c,const float* pcm,size_t count);
/* prompt: NULL for default, otherwise <=16384 UTF-8 bytes; max_new_tokens: 1..2048. */
A8_API int a8_finish(a8_context* c,const char* prompt,int max_new_tokens);
/* Query size with NULL/0. Both getters require a completed single generation;
 * no result after a reset, an execution failure, or batch generation (STATE). */
A8_API int a8_get_tokens(a8_context* c,int32_t* dst,size_t capacity,size_t* required);
/* required includes trailing NUL for text; token_bytes reports raw byte count, with no terminator. */
A8_API int a8_get_text(a8_context* c,char* dst,size_t capacity,size_t* required);
A8_API int a8_token_bytes(a8_context* c,int32_t id,char* dst,size_t capacity,size_t* required);
/* Single-audio processor ABI1. All handles are process/provider-local; one caller
 * per handle, with explicit synchronization against close. Inputs are borrowed
 * until the synchronous call returns. Getter counts are elements, text sizes
 * include NUL. NULL/0 queries report required size; insufficient capacity writes
 * no elements. prepare replaces the processor's retained arrays only on success.
 * Model-specific processor work executes on a worker bound by the Runtime.
 * Resampling overshoot is permitted for finite model-rate PCM; Python validates
 * source PCM normalization before resampling. JSON hotword input is a normalized array of UTF-8 strings, <=1 MiB; compiled
 * plans are <=16 MiB. These resource limits never silently truncate data. */
typedef struct a8_processor a8_processor_t;
A8_API int a8_processor_create(const char*,pandora_runtime_t*,a8_processor_t**);
/* Reuse a live model's immutable tokenizer/frontend and Runtime. The output slot
 * must point to NULL. The owned processor has independent preparation results
 * and stays usable after the model is destroyed. Synchronize with calls on the
 * model through this call and use a8_processor_destroy_owned for the output. */
A8_API int a8_processor_create_from_model(a8_context* model, a8_processor_t** out);
A8_API int a8_processor_destroy_owned(a8_processor_t**);
A8_API int a8_processor_prepare(a8_processor_t*,const float*,size_t,const char*);
A8_API int a8_processor_get_ids(a8_processor_t*,int32_t*,size_t,size_t*);
A8_API int a8_processor_get_features(a8_processor_t*,float*,size_t,size_t*);
A8_API int a8_processor_encode(a8_processor_t*,const char*,int32_t*,size_t,size_t*);
A8_API int a8_processor_special_ids(a8_processor_t*,int32_t*,size_t,size_t*);
A8_API int a8_processor_symbol(a8_processor_t*,int32_t,char*,size_t,size_t*);
enum { A8_DECODE_SKIP_SPECIAL = 1, A8_DECODE_ALLOW_INCOMPLETE_TAIL = 2 };
/* Pass A8_DECODE_ALLOW_INCOMPLETE_TAIL only after a length-limited generation. */
A8_API int a8_processor_decode(a8_processor_t*,const int32_t*,size_t,uint32_t,char*,size_t,size_t*);
A8_API int a8_processor_hotword_plan(a8_processor_t*,const char*,int64_t,double,double,char*,size_t,size_t*);
/* ABI1: real input IDs include prompt, attention_mask is NULL (all ones) or
 * count input_count; features are contiguous [128,frames] float32, frames1..3000.
 * Use a8_generate_batch for batched inference. Invalid structure/shape/template
 * fails before reset/NPU.
 * hotword_plan is NULL for off or a compiled native JSON plan; no Python callbacks.
 * max_new_tokens1..2048; full reservation must fit CL2048.
 * get_tokens returns new tokens only; caller prepends validated input IDs.
 * All request buffers may be freed after return. Output copies survive close. */
typedef struct {
 uint32_t struct_size,abi_version;
 const int32_t* input_ids;
 size_t input_count;
 const int32_t* attention_mask;
 const float* input_features;
 uint32_t frames,max_new_tokens;
 const char* hotword_plan;
 uint32_t reserved[2];
} a8_generation_request;
#define A8_GENERATION_ABI_VERSION 1
A8_API int a8_generate_prepared(a8_context*,const a8_generation_request*);
/* Batch ABI 2: 2..32 unpadded prepared rows using a8_generation_request.
 * Initialize struct_size to sizeof(a8_batch_request), abi_version to
 * A8_BATCH_ABI_VERSION, and reserved fields to zero. The rows and their
 * buffers are borrowed until return. All rows validate before execution.
 * out must point to NULL. Failure leaves it NULL. An execution failure
 * requires close/recreate. There is no partial result.
 * The owned result survives model close and is read in input order.
 * Per-row get returns generated IDs only, with reason 1=EOS or 2=budget. */
typedef struct a8_batch_result a8_batch_result;
#define A8_BATCH_ABI_VERSION 2
typedef struct {
 uint32_t struct_size,abi_version;
 const a8_generation_request* rows;
 size_t count;
 uint32_t reserved[2];
} a8_batch_request;
A8_API int a8_generate_batch(a8_context*,const a8_batch_request*,a8_batch_result**);
A8_API int a8_batch_result_get(const a8_batch_result*,size_t,int32_t*,size_t,size_t*,uint32_t*);
A8_API int a8_batch_result_destroy_owned(a8_batch_result**);

/* reason:1=EOS,2=budget exhausted. Never returns a successful unknown end. */
A8_API int a8_get_generation_status(a8_context*,uint32_t* reason,size_t* generated);

A8_API const char* a8_last_error(void);
#ifdef __cplusplus
}
#endif
