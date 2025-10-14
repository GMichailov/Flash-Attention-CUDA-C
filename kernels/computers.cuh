#pragma once

#include "utils.cuh"

// TODO: Bring in missing vars
/*
template<int DHEAD, int BLOCK_ROWS, int ROWS_PER_WARP>
__device__ __forceinline__ void singleLoaderMhaComputeWarp(
    float* __restrict__ (&smem)[2],
    pipe_t pipeQ, pipe_t pipeK, pipe_t pipeV, auto& block,
    int batchSize, int numHeads, int seqLenQ, int seqLenK
) {
    constexpr int tileSize = DHEAD * BLOCK_ROWS;
    constexpr int QFragmentSize = tileSize / WARP;

    // Create partitions per Q row
    auto warp = cg::tiled_partition<WARP>(block);
    auto rowGroup = cg::tiled_partition<WARP / ROWS_PER_WARP>(warp);

    // Set smem pointers needed for calculations
    float* smemQ[2];
    float* smemK[2];
    float* smemV[2];
    float* smemL;
    float* smemM;
    float QFrag[QFragmentSize];
    setLoaderSmemPointers(smemQ, smemK, smemV, KTileSize, BLOCK_ROWS);
    oneLoaderSetCalculatorAdditionalSmemPointers(smemL, smemM, tileSize, BLOCK_ROWS);

    // Allocate/Store reused data.
    int buf = 0;
    float score;
    float newMax;
    float newL;
    // Begin Iterating through tiles.
    for(int absoluteQRow = 0; absoluteQRow < seqLenQ * batchSize * numHeads; absoluteQRow += BLOCK_ROWS) {
        int relativeKvRow = rowGroup.thread_rank() / (WARP / ROWS_PER_WARP);
        for (int absoluteKvRow = 0; absoluteKvRow < seqLenK * batchSize * numHeads; absoluteKvRow += BLOCK_ROWS) {
            pipeK.consumer_wait();
            // Each rowGroup handles its calculations
            if (absoluteKvRow + relativeKvRow > absoluteQRow) {
                score = -INFINITY;
            } else {
                computeRowNoncausalAttentionScore<ROWS_PER_WARP>(QFrag, smemK[buf], kvRow, laneId, scale, D_HEAD, QFragmentSize, rowGroup);
            }
            if (!rowGroup.thread_rank()) {
                // Pass correct Q row.
                rowSoftmax(&smemM, &smemL, , score, newMax, newL);
            }
            newMax = rowGroup.shfl(newMax, 0);
            newL = rowGroup.shfl(newL, 0);
            rowGroup.sync();
            pipeV.consumer_wait();
            multiplyVStoreO();
            buf ^= 1;
        }
    }
}
*/

template<int D_HEAD, int Q_TILE_ROWS, int KV_TILE_ROWS>
__device__ __forceinline__ void twoLoaderMhaComputeWarp(
    auto& block, int batchSize, int numHeads, int seqLen, float scale, bool is_causal,
    auto& pipeQ, auto& pipeK, auto& pipeV, auto& pipeO
) {
    constexpr int qTileElements = D_HEAD * Q_TILE_ROWS;
    constexpr int kvTileElements = D_HEAD * KV_TILE_ROWS;

    // Create partitions per Q row
    auto warp = cg::tiled_partition<WARP>(block);
    auto group = cg::tiled_partition<WARP / KV_TILE_ROWS>(warp);

    // Set smem pointers needed for calculations
    float* smemQ[2];
    float* smemK[2];
    float* smemV[2];
    float* smemO;
    setComputerSmemPointers(smemQ, smemK, smemV, smemO, qTileElements, kvTileElements);

    // Allocate/Store reused data.
    uint8_t bufQ = 0;
    float score;
    float running_max;
    float running_l;

    // Grab Q and stream KV against it to be able to store O stuff per warp and only keep a Q_TILE_ROWS tile for O.
    for (int globalQRow = 0; globalQRow < batchSize * numHeads * seqLen; globalQRow += Q_TILE_ROWS) {
        score = 0.0f;
        running_max = -INFINITY;
        running_l = 0.0f;
        uint8_t bufKV = 0;
        pipeQ.producer_acquire();
        pipeQ.producer_commit();
        pipeQ.consumer_wait();
        pipeO.producer_acquire();
        for (int globalKVRow = 0; globalKVRow < batchSize * numHeads * seqLen; globalKVRow += KV_TILE_ROWS) {
            pipeK.producer_acquire();
            pipeK.producer_commit();
            pipeK.consumer_wait();
            if (is_causal && (globalKVRow + group.meta_group_rank() > globalQRow + warp.meta_group_rank())) score = -INFINITY;
            else computeAttentionScore<Q_TILE_ROWS, KV_TILE_ROWS, D_HEAD>(smemQ[bufQ], smemK[bufKV], scale, warp, group, score);
            pipeK.consumer_release();
            float curr_max;
            float curr_l;
            group.sync();
            // Sum group to get score and then find max score for warp.
            score = cg::reduce(group, score, cg::plus<float>());
            curr_max = cg::reduce(warp, score, cg::greater<float>());
            // Sum up scores held by group leaders in warp and find max simultaneously.
            curr_l = (group.thread_rank() == 0) ? expf(score - curr_max) : 0.0f;
            unsigned mask = groupLeaderMask(group.size());
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                curr_l += __shfl_down_sync(mask, curr_l, offset);
            }
            // Caculate the actual l itself in thread 0.
            if (warp.thread_rank() == 0) curr_l = expf(running_max - fmaxf(curr_max, running_max)) * running_l + expf(curr_max - fmaxf(curr_max, running_max)) * curr_l;
            // Broadcast out the newly calculated l to all warp threads.
            running_l = __shfl_sync(0xFFFFFFFF, curr_l, 0);
            running_max = fmaxf(curr_max, running_max);
            // Multiply against V
            float weight = expf(score - running_max) / running_l;
            pipeV.producer_acquire();
            pipeV.producer_commit();
            pipeV.consumer_wait();
            // Create mask so that thread X of each group in the warp sees each other.
            threadRankMask(group.size(), group.thread_rank(), mask);
            // For each float in fragment, accumulate into thread X of group 0 which will write to corresponding smemO index.
            for (int idx = 0; idx < (D_HEAD / group.size()); ++idx) {
                float out = weight * smemV[bufKV][static_cast<int>(warp.meta_group_rank()) * D_HEAD + static_cast<int>(group.thread_rank()) * idx];
                for (int offset = group.size(); offset < 32; offset += group.size()) {
                    out += __shfl_down_sync(mask, out, offset);
                }
                if (group.meta_group_rank() == 0) {
                    if (globalKVRow == 0) smemO[static_cast<int>(warp.meta_group_rank()) * D_HEAD + static_cast<int>(group.thread_rank()) * idx] = out;
                    else smemO[warp.meta_group_rank() * D_HEAD + group.thread_rank() * idx] += out;
                }
            }
            pipeV.consumer_release();
            bufKV ^= 1;
        }
        pipeQ.consumer_release();
        pipeO.producer_commit();
        pipeO.consumer_wait();
        pipeO.consumer_release();
        bufQ ^= 1;
    }
}