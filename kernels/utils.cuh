#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda/pipeline>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define WARP 32
#define FLOAT_SIZE 4

using pipe_t = cuda::pipeline<cuda::thread_scope_block>;

template<int ROWS_PER_WARP, int D_HEAD>
__global__ __forceinline__ int computeRowNoncausalAttentionScore(
    const float* smemQBuf, const float* smemKBuf,
    int relativeQRow, int relativeKvRow, float scale, int QFragmentSize,
    cg::thread_block_tile<WARP / ROWS_PER_WARP>& rowGroup
) {
    float partialDotProduct = 0.0f;
    // QFragmentSize bc is equal due to being along d_head for both.
    const float* qRowPtr = &smemQBuf[relativeQRow * D_HEAD + rowGroup.thread_rank() * QFragmentSize];
    const float* kRowPtr = &smemKBuf[relativeKvRow * D_HEAD + rowGroup.thread_rank() * QFragmentSize];
    #pragma unroll
    for (int i = 0; i < QFragmentSize; i++) {
        partialDotProduct += qRowPtr[i] * kRowPtr[i];
    }
    float score = cg::reduce(rowGroup, partialDotProduct, cg::plus<float>()) * scale;
    return score;
}


__global__ __forceinline__ void rowSoftmax(
    float* __restrict__ smemM, float* __restrict__ smemL, 
    int qRow, float score, float& newMax, float& newL
) {
    newMax = fmaxf(smemM[qRow], score);
    newL = smemL[qRow] * expf(smemM[qRow] - newMax) + expf(score - newMax);
}


template<int ROWS_PER_WARP>
__global__ __forceinline__ void multiplyVStoreO(
    float* __restrict__ smemV, float* __restrict__ O, float* __restrict__ OFrag,
    float* __restrict__ smemL, float* __restrict__ smemM, 
    int QFragmentSize, int qIdx, int qRow, int kvRow, const float& score,
    const float& newMax, const float& newL,
    cg::thread_block_tile<WARP / ROWS_PER_WARP>& rowGroup
) {
    float* vPtr = &smemV[kvRow + rowGroup.thread_rank() * QFragmentSize];
    float* oPtr = O[qIdx];
    float weight = expf(score - newMax) / newL;
    float rescale = smemL[qRow] * expf(smemM[qRow] - newMax) / newL;
    #pragma unroll
    for(int i = 0; i < QFragmentSize; ++i) {
        oPtr[i] = oPtr[i] * rescale + weight * vPtr[i];
    }
}

__global__ __forceinline__ void updateML(
    float* __restrict__ smemL, float* __restrict__ smemM,
    int qRow, const float& newL, const float& newMax
) {
    smemL[qRow] = newL;
    smemM[qRow] = newMax;
}

