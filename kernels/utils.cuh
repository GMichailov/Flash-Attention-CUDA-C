#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda/pipeline>
#include <cuda_runtime_api.h>

namespace cg = cooperative_groups;

#define WARP 32
#define FLOAT_SIZE 4

using pipe_t = cuda::pipeline<cuda::thread_scope_block>;

template<int ROWS_PER_WARP>
__global__ __forceinline__ int computeRowNoncausalAttentionScore(
    const float* QFrag, const float* smemKBuf,
    int kvrow, int laneId, float scale, int D_HEAD, int QFragmentSize,
    cg::thread_block_tile<WARP / ROWS_PER_WARP>& rowGroup
) {
    float partialDotProduct = 0.0f;
    // QFragmentSize bc is equal due to being along d_head for both.
    const float* kRowPtr = &smemK[buf][kvRow * D_HEAD + laneId * QFragmentSize];
    #pragma unroll
    for (int i = 0; i < QFragmentSize; i++) {
        partialDotProduct += QFrag[i] * kRowPtr[i];
    }
    float score = cg::reduce(rowGroup, partialDotProduct, cg::plus<float>()) * scale;
    return score;
}


__global__ __forceinline__ void rowSoftmax(
    float* __restrict__ smemM, float* __restrict__ smemL, 
    int qIdx, float score
) {
    float newMax = fmaxf(smemM[qIdx], score);
    smemL[qIdx] = (smemL[qIdx] + 1) * expf(smemM[qIdx] - newMax);
    smemM[qIdx] = newMax;
}


template<int ROWS_PER_WARP>
__global__ __forceinline__ void multiplyVStoreO(
    const float* vRowPtr, const float* oRowPtr, float* OFrag, float score,
    int QFragmentSize,
    cg::thread_block_tile<WARP / ROWS_PER_WARP>& rowGroup
) {
    // Calculate V and O row ptrs here.
    // Scale with softmax trick.
    #pragma unroll
    for(int i = 0; i < QFragmentSize; ++i) {
        oRowPtr[i] = score * vRowPtr[i];
    }
}