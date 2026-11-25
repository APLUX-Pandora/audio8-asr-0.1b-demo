#pragma once
#include "audio8.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Frozen ABI 1 compatibility. Use audio8.h for new integrations. */
/* Single-audio generated-token streaming, ABI1. Same request/validation/results as
 * a8_generate_prepared. on_token executes synchronously on the model worker for
 * each new token, including EOS; it never receives prompt tokens. Return 0 to
 * continue; nonzero aborts with STATE and requires close/recreate. No exceptions
 * may escape a C callback. Do not reenter this context, destroy it, or wait for
 * its caller from the callback. Callback/user_data are borrowed until return.
 * This function does not invoke a completion callback: its return code is final.
 * Initialize size/version=sizeof/1 and reserved fields to zero. */
typedef int (*a8_token_callback)(int32_t token_id, void* user_data);
typedef struct {
 uint32_t struct_size,abi_version;
 a8_token_callback on_token;
 void* user_data;
 uint32_t reserved[2];
} a8_stream_options;
A8_API int a8_generate_prepared_stream(a8_context*,const a8_generation_request*,const a8_stream_options*);

/* ABI 1 calls use one vector per generation step, in input order, with
 * EOS included and PAD after a row completes. The first nonzero callback
 * result aborts the batch and requires recreation of the model. */
typedef int (*a8_batch_token_callback)(const int32_t*,size_t,void*);
typedef struct {
 uint32_t struct_size,abi_version;
 const a8_generation_request* rows;
 size_t count;
 a8_batch_token_callback on_tokens;
 void* user_data;
 uint32_t reserved[2];
} a8_batch_request_v1;

static inline int a8_generate_batch_v1(a8_context* model,
                                       const a8_batch_request_v1* request,
                                       a8_batch_result** out) {
    return a8_generate_batch(model, (const a8_batch_request*)(const void*)request, out);
}

#ifdef __cplusplus
}
#endif
