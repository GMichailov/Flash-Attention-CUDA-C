#pragma once

#include "utils.cuh"
#include "loaders.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <cmath>
#include <cuda/pipeline>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define WARP 32
#define FLOAT_SIZE 4

using pipe_t = cuda::pipeline<cuda::thread_scope_block>;

__shared__ pipe_t pipeQ;
__shared__ pipe_t pipeK;
__shared__ pipe_t pipeV;


// Kernel that uses only 1 warp for loading from HBM into SRAM (High compute needs so maybe for FP32)
template<int DHEAD, int BLOCK_Q_ROWS, int BLOCK_KV_ROWS, int ROWS_PER_WARP>
__global__ void oneLoaderMhaFlashAttentionKernel(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
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
);