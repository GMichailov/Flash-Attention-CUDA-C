#pragma once

#include "utils.cuh"

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
    float score, weight, scaling_factor;
    float running_max, running_l;
    float prev_max, prev_l;
    float curr_max, curr_l;
    unsigned mask;

    // Grab Q and stream KV against it to be able to store O stuff per warp and only keep a Q_TILE_ROWS tile for O.
    for (int globalQRow = 0; globalQRow < batchSize * numHeads * seqLen; globalQRow += Q_TILE_ROWS) {
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

            score = computeTileScore<D_HEAD, Q_TILE_ROWS, KV_TILE_ROWS>(smemQ[bufQ], smemV[bufKV], scale, is_causal, globalQRow, globalKVRow, warp, group);

            pipeK.consumer_release();
            group.sync();
            
            updateSoftmaxState(score, mask, prev_max, prev_l, running_max, running_l, curr_max, curr_l, weight, scaling_factor, warp, group);
            
            pipeV.producer_acquire();
            pipeV.producer_commit();
            pipeV.consumer_wait();
            
            multiplyVAccumulateO<D_HEAD>(smemV[bufKV], smemO, warp, group, mask, weight, scaling_factor, globalKVRow);

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