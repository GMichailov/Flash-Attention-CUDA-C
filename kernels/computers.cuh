#pragma once

#include "utils.cuh"

// TODO: Bring in missing vars
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


template<int D_HEAD, int Q_TILE_ROWS, int KV_TILE_ROWS>
__device__ __forceinline__ void twoLoaderMhaComputeWarp(
    auto& block, int batchSize, int numHeads, int seqLen
) {
    constexpr int qTileElements = D_HEAD * Q_TILE_ROWS;
    constexpr int kvTileElements = D_HEAD * KV_TILE_ROWS;
    constexpr int perThreadfragmentSizeQ = qTileElements / WARP;
    constexpr int perThreadfragmentSizeKV = kvTileElements / WARP;

    // Create partitions per Q row
    auto warp = cg::tiled_partition<WARP>(block);
    auto group = cg::tiled_partition<WARP / KV_TILE_ROWS>(warp);

    // Set smem pointers needed for calculations
    float* smemQ[2];
    float* smemK[2];
    float* smemV[2];
    float* smemO[2];
    setComputerSmemPointers(smemQ, smemK, smemV, smemO, qTileElements, kvTileElements);

    // Allocate/Store reused data.
    uint8_t bufQ = 0;
    uint8_t bufKV = 0;
    float score;
    float newMax;
    float newL;




    //==========================================================================================================================
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