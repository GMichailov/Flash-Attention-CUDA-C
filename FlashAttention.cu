#include <cuda.h>
#include <cuda_runtime.h>


/*
 * Discoveries and Notes:
 * - In regular LLMs, seq_len_q and kv are the same length because they accept the same input, however
 *   in seq2seq tasks like translating languages, they can be different.
 * - The strides tell how far between Q[batch, num_heads, seq_len=0, d_head=0] and 
 *   Q[batch, num_heads + 1, seq_len=0, d_head=0] which allows packed layouts and can 
 *   handle multihead attn without forcing into contiguous layout => No unnecessary copies + reorders.
 *   TLDR: Tells how far to jump once flattened.
 */
__global__ void FlashAttentionV1(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V, float* __restrict__ O,
    float* __restrict__ L, float* __restrict__ M,
    int batch_size, 
    int num_heads, 
    int seq_len_q,
    int seq_len_k, 
    int d_head,
    int stride_q,
    int stride_k,
    int stride_v,
    int stride_o,
    float scale // Compute 1/root(d_head) on CPU.
)
{

}


// Implement causal separate or later.