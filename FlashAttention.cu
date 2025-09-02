#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <cmath>

#define HEAD 64
#define WARP 32
#define FRAG HEAD/WARP

/*
 * Discoveries and Notes:
 * - In regular LLMs, seq_len_q and kv are the same length because they accept the same input, however
 *   in seq2seq tasks like translating languages, they can be different.
 * - The strides tell how far between Q[batch, num_heads, seq_len=0, d_head=0] and 
 *   Q[batch, num_heads + 1, seq_len=0, d_head=0] which allows packed layouts and can 
 *   handle multihead attn without forcing into contiguous layout => No unnecessary copies + reorders.
 *   TLDR: Tells how far to jump once flattened.
 * - Each block handles a group of query rows across one head and one batch.
 */
__global__ void FlashAttentionV1(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V, float* __restrict__ O,
    float* __restrict__ L, float* __restrict__ M,
    int batch_size, 
    int num_heads, 
    int seq_len_q,
    int seq_len_k, 
    int d_head,
    int stride_batch_q,
    int stride_batch_k,
    int stride_batch_v,
    int stride_batch_o,
    int stride_head_q,
    int stride_head_k,
    int stride_head_v,
    int stride_head_o,
    float scale, // Compute 1/root(d_head) on CPU.
    int BLOCK_Q_ROWS, // Offload responsibility to CPU function to calculate biggest amount that would fit in shared memory.
    int BLOCK_KV_ROWS
)
{
    // Step 1: Use BlockIdx, ThreadIdx, Block_q_rows to load into register memory.
    int warpId = threadIdx.x / WARP;
    int laneId = threadIdx.x % WARP;
    int qTileStart = blockIdx.x * BLOCK_Q_ROWS;
    int qRow = qTileStart + warpId;
    if (qRow >= seq_len_q) {
        return;
    }
    const float* qPtr = Q + blockIdx.z * stride_batch_q + blockIdx.y * stride_head_q + qRow * d_head + laneId * FRAG;

    float registerMemory[FRAG];
    #pragma unroll
    for (int i = 0; i < FRAG; ++i){
        registerMemory[i] = qPtr[i];
    }

    // Step 2: Load K and V block into shared memory in a similar way where each thread pulls some elements in.
    extern __shared__ float smem[];
    float* kSmem = smem;
    float* vSmem = smem + BLOCK_KV_ROWS * d_head;

    float mWarp = -INFINITY;
    float lWarp = 0.0f;

    for (int kv_tile_start = 0; kv_tile_start < seq_len_k; kv_tile_start += BLOCK_KV_ROWS) {
        int rowsInTile = std::min(BLOCK_KV_ROWS, seq_len_k - kv_tile_start);
        const float* kTilePtr = K + blockIdx.z * stride_batch_k + blockIdx.y * stride_head_k + kv_tile_start * d_head;
        const float* vTilePtr = V + blockIdx.z * stride_batch_v + blockIdx.y * stride_head_v + kv_tile_start * d_head;
        int total_elements = rowsInTile * d_head;
        #pragma unroll
        for (int idx = threadIdx.x; idx < total_elements; idx += blockDim.x) {
            kSmem[idx] = kTilePtr[idx];
            vSmem[idx] = vTilePtr[idx];
        }
        __syncthreads();

        // Step 3: Compute QK^T
        for (int kvRow = 0; kvRow < rowsInTile; ++kvRow) {
            const float* kRowPtr = &kSmem[kvRow * d_head + laneId * FRAG];
            float partialDotProduct = 0.0f;
            #pragma unroll
            for (int i = 0; i < FRAG; i++) {
                partialDotProduct += registerMemory[i] * kRowPtr[i];
            }

            // Warp reduction for summing across all lanes (row sum).
            for (int offset = WARP / 2; offset > 0; offset /= 2) {
                partialDotProduct += __shfl_down_sync(0xffffffff, partialDotProduct, offset);
            }

            float score = __shfl_sync(0xffffffff, partialDotProduct, 0) * scale;

            // Step 4: Apply Softmax
        }


        
        // Repear steps 2-4 until done.
    }


    
}
 

// Implement causal separate or later.