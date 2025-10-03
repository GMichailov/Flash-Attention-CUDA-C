#include "FlashAttention.cuh"

// Kernel that uses only 1 warp for loading from HBM into SRAM (High compute)
template<int DHEAD, int BLOCK_Q_ROWS, int BLOCK_KV_ROWS, int ROWS_PER_WARP>
__global__ void oneLoaderMhaFlashAttentionKernel(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V, float* __restrict__ O,
    const float* __restrict__ L, const float* __restrict__ M,
    int batchSize, int numHeads,
    int seqLenQ, int seqLenK,
    int strideBatchQ, int strideBatchK, int strideBatchV, int strideBatchO,
    int strideHeadQ, int strideHeadK, int strideHeadV, int strideHeadO,
    float scale, 
    int BLOCK_KV_ROWS, bool is_causal
)
{
    extern __shared__ float smem[];
    auto block = cg::this_thread_block();
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block> pipeStateQ;
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block> pipeStateK;
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block> pipeStateV;

    auto pipeQ = cuda::make_pipeline(block, &pipeStateQ);
    auto pipeK = cuda::make_pipeline(block, &pipeStateK);
    auto pipeV = cuda::make_pipeline(block, &pipeStateV);

    int warpId = threadIdx.x / WARP;
    int laneId = threadIdx.x % WARP;
    // Split off loader warps
    if (warpId == 0) {
        singleLoaderWarp(smem, pipeQ, pipeK, pipeV, block, laneId);
    } else {
        singleLoaderMhaComputeWarp(smem, pipeQ, pipeK, pipeV, block, laneId);
    }
}