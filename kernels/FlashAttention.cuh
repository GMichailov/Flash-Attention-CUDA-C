#pragma once

#include "utils.cuh"
#include "loaders.cuh"
#include "computers.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <cmath>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

/*
// Kernel that uses only 1 warp for loading from HBM into SRAM (High compute needs so maybe for FP32)
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
);

// Kernel that uses 2 warps for loading from HBM into SRAM 
// Warp 0 does Q and then V.
// Warp 1 does K.
template<int DHEAD, int BLOCK_Q_ROWS, int BLOCK_KV_ROWS, int ROWS_PER_WARP>
__global__ void twoLoaderMhaFlashAttentionKernel(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    const float* __restrict__ L, const float* __restrict__ M,
    int batchSize, int numHeads,
    int seqLenQ, int seqLenK,
    int strideBatchQ, int strideBatchK, int strideBatchV, int strideBatchO,
    int strideHeadQ, int strideHeadK, int strideHeadV, int strideHeadO,
    float scale, 
    int BLOCK_KV_ROWS, bool is_causal
);

// Kernel that uses 3 warps for loading from HBM into SRAM
// Warps 0-2 do Q, K, V respectively.
template<int DHEAD, int BLOCK_Q_ROWS, int BLOCK_KV_ROWS, int ROWS_PER_WARP>
__global__ void threeLoaderMhaFlashAttentionKernel(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    const float* __restrict__ L, const float* __restrict__ M,
    int batchSize, int numHeads,
    int seqLenQ, int seqLenK,
    int strideBatchQ, int strideBatchK, int strideBatchV, int strideBatchO,
    int strideHeadQ, int strideHeadK, int strideHeadV, int strideHeadO,
    float scale, 
    int BLOCK_KV_ROWS, bool is_causal
);*/

template<int D_HEAD, int Q_TILE_ROWS, int KV_TILE_ROWS, typename scalar_t>
__global__ void twoLoaderMhaFlashAttentionKernel(
    const scalar_t* __restrict__ Q, const scalar_t* __restrict__ K, const scalar_t* __restrict__ V, scalar_t* __restrict__ O,
    int batchSize, int numHeads, int seqLen, scalar_t scale, bool is_causal
)
{
    auto block = cg::this_thread_block();
    int warpId = threadIdx.x / WARP;
    int numWarps = blockDim.x / WARP;
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pipeStateQ;
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pipeStateK;
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pipeStateV;
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pipeStateO;
    auto pipeQ = cuda::make_pipeline(block, &pipeStateQ);
    auto pipeK = cuda::make_pipeline(block, &pipeStateK);
    auto pipeV = cuda::make_pipeline(block, &pipeStateV);
    auto pipeO = cuda::make_pipeline(block, &pipeStateO);
    // Split off loader warps
    if (warpId < numWarps - 2) {
        twoLoaderMhaComputeWarp<D_HEAD, Q_TILE_ROWS, KV_TILE_ROWS, scalar_t>(block, batchSize, numHeads, seqLen, scale, is_causal, pipeQ, pipeK, pipeV, pipeO);
    } else if (warpId == numWarps - 2) {
        qvLoaderWarp<D_HEAD, Q_TILE_ROWS, KV_TILE_ROWS, scalar_t>(Q, V, block, batchSize, numHeads, seqLen, pipeQ, pipeK, pipeV, pipeO);
    } else {
        koLoaderWarp<D_HEAD, Q_TILE_ROWS, KV_TILE_ROWS, scalar_t>(K, O, block, batchSize, numHeads, seqLen, pipeQ, pipeK, pipeV, pipeO);
    }
}