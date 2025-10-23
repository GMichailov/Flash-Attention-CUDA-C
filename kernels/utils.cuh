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

template<int Q_TILE_ROWS, int KV_TILE_ROWS, int D_HEAD> 
__device__ __forceinline__ float computeAttentionScore(
    const float* smemQPtr, const float* smemKPtr,
    float scale, const auto& warp, const auto& group
) {
    float score = 0.0f;
    uint8_t fragmentSize = D_HEAD / group.size();
    smemQPtr += warp.meta_group_rank() * D_HEAD + group.thread_rank() * fragmentSize;
    smemKPtr += group.meta_group_rank() * D_HEAD + group.thread_rank() * fragmentSize;
    #pragma unroll
    for (int i = 0; i < fragmentSize; ++i) {
        score += smemQPtr[i] * smemKPtr[i];
    }
    group.sync();
    score = cg::reduce(group, score, cg::plus<float>()) * scale;
    return score;
}


template<int D_HEAD, int Q_TILE_ROWS, int KV_TILE_ROWS>
__device__ __forceinline__ float computeTileScore(
    const float* smemQBuf, const float* smemKBuf,
    float& scale, bool is_causal,
    int globalQRow, int globalKVRow,
    const auto& warp, const auto& group
) {
    if (is_causal && (globalKVRow + group.meta_group_rank() > globalQRow + warp.meta_group_rank())) return -INFINITY;
    else return computeAttentionScore<Q_TILE_ROWS, KV_TILE_ROWS, D_HEAD>(smemQBuf, smemKBuf, scale, warp, group);
}



__device__ __forceinline__ void groupLeaderMask(int group_size, unsigned& mask) {
    mask = 0;
    #pragma unroll
    for (uint8_t lane = 0; lane < 32; lane += group_size) {
        mask |= (1u << lane);
    }
}


__device__ __forceinline__ void updateSoftmaxState(
    float score, unsigned& mask,
    float& prev_max, float& prev_l, float& running_max, float& running_l, float& curr_max, float& curr_l, float& weight, float& scaling_factor,
    const auto& warp, const auto& group
) {
    prev_max = running_max;
    prev_l = running_l;
    // Sum group to get score and then find max score for warp.
    curr_max = cg::reduce(warp, score, cg::greater<float>());
    // Sum up scores held by group leaders in warp and find max simultaneously.
    curr_l = (group.thread_rank() == 0) ? expf(score - curr_max) : 0.0f;
    groupLeaderMask(group.size(), mask);
    #pragma unroll
    for (int offset = group.size(); offset < WARP; offset += group.size()) {
        curr_l += __shfl_down_sync(mask, curr_l, offset);
    }
    // Caculate the actual l itself in thread 0.
    if (warp.thread_rank() == 0) curr_l = expf(running_max - fmaxf(curr_max, running_max)) * running_l + expf(curr_max - fmaxf(curr_max, running_max)) * curr_l;
    // Broadcast out the newly calculated l to all warp threads.
    running_l = __shfl_sync(0xFFFFFFFF, curr_l, 0);
    running_max = fmaxf(curr_max, running_max);
    weight = expf(score - running_max) / running_l;
    scaling_factor = expf(prev_max - running_max) * (prev_l / running_l);
}


__device__ __forceinline__ void threadRankMask(int group_size, int group_thread_rank, unsigned& mask) {
    mask = 0;
    #pragma unroll
    for (uint8_t i = group_thread_rank; i < 32; i += group_size) {
        mask |= (1u << i);
    }
}


template<int D_HEAD>
__device__ __forceinline__ void multiplyVAccumulateO(
    float* smemVBuf, float* smemO,
    const auto& warp, const auto& group, 
    unsigned& mask, const float& weight, float& scaling_factor, int& globalKVRow
) {
    // Create mask so that thread X of each group in the warp sees each other.
    threadRankMask(group.size(), group.thread_rank(), mask);
    // For each float in fragment, accumulate into thread X of group 0 which will write to corresponding smemO index.
    for (int idx = 0; idx < (D_HEAD / group.size()); ++idx) {
        float out = weight * smemVBuf[group.meta_group_rank() * D_HEAD + group.thread_rank() * (D_HEAD / group.size()) + idx];
        for (int offset = group.size(); offset < WARP; offset += group.size()) {
            out += __shfl_down_sync(mask, out, offset);
        }
        if (group.meta_group_rank() == 0) {
            int oIdx = warp.meta_group_rank() * D_HEAD + group.thread_rank() * (D_HEAD / group.size()) + idx;
            if (globalKVRow == 0) smemO[oIdx] = out;
            else smemO[oIdx] = scaling_factor * smemO[oIdx] + out;
        }
    }
}
