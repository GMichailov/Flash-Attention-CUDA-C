#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda/pipeline>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

#define WARP 32
#define FLOAT_SIZE 4

using pipe_t = cuda::pipeline<cuda::thread_scope_block>;

template<int ROWS_PER_WARP, int D_HEAD>
__device__ __forceinline__ int computeRowNoncausalAttentionScore(
    const float* smemQBuf, const float* smemKBuf,
    int relativeRow, float scale, int fragmentSize,
    cg::thread_block_tile<WARP / ROWS_PER_WARP>& rowGroup
) {
    float partialDotProduct = 0.0f;
    int idx = relativeRow * D_HEAD + rowGroup.thread_rank() * fragmentSize;
    const float* qRowPtr = &smemQBuf[idx];
    const float* kRowPtr = &smemKBuf[idx];
    #pragma unroll
    for (int i = 0; i < fragmentSize; i++) {
        partialDotProduct += qRowPtr[i] * kRowPtr[i];
    }
    float score = cg::reduce(rowGroup, partialDotProduct, cg::plus<float>()) * scale;
    return score;
}


template<int Q_TILE_ROWS, int KV_TILE_ROWS, int D_HEAD> 
__device__ __forceinline__ void computeAttentionScore(
    float* smemQPtr, float* smemKPtr,
    const float scale, auto& warp, auto& group, float& score
) {
    uint8_t fragmentSize = D_HEAD / group.size();
    smemQPtr += warp.meta_group_size() * D_HEAD + group.thread_rank() * fragmentSize;
    smemKPtr += group.meta_group_rank() * D_HEAD + group.thread_rank() * fragmentSize;
    #pragma unroll
    for (int i = 0; i < fragmentSize; ++i) {
        score += smemQPtr[i] * smemKPtr[i];
    }
    group.sync();
    score = cg::reduce(group, score, cg::plus<float>()) * scale;
}


__device__ __forceinline__ unsigned groupLeaderMask(int group_size) {
    unsigned mask = 0;
    #pragma unroll
    for (uint8_t lane = 0; lane < 32; lane += group_size) {
        mask |= (1u << lane);
    }
    return mask;
}


__device__ __forceinline__ void threadRankMask(int group_size, int group_thread_rank, unsigned& mask) {
    mask = 0;
    #pragma unroll
    for (uint8_t i = group_thread_rank; i < 32; i += group_size) {
        mask |= (1u << i);
    }
}

/*
__device__ __forceinline__ void rowSoftmax(
    float* __restrict__ smemM, float* __restrict__ smemL, 
    int qRow, float score, float& newMax, float& newL
) {
    newMax = fmaxf(smemM[qRow], score);
    newL = smemL[qRow] * expf(smemM[qRow] - newMax) + expf(score - newMax);
}


template<int ROWS_PER_WARP>
__device__ __forceinline__ void multiplyVStoreO(
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
*/
__device__ __forceinline__ void updateML(
    float* __restrict__ smemL, float* __restrict__ smemM,
    int qRow, const float& newL, const float& newMax
) {
    smemL[qRow] = newL;
    smemM[qRow] = newMax;
}

