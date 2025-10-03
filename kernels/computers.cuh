#pragma once

#include "utils.cuh"

// TODO: Bring in missing vars
template<int DHEAD, int BLOCK_Q_ROWS, int BLOCK_KV_ROWS, int ROWS_PER_WARP>
__device__ __forceinline__ void singleLoaderMhaComputeWarp(float* __restrict__ (&smem)[2], pipe_t pipeQ, pipe_t pipeK, pipe_t pipeV, auto block, int laneId) {
    constexpr int QTileSize = DHEAD * BLOCK_Q_ROWS;
    constexpr int QFragmentSize = QTileSize / WARP;
    constexpr int KTileSize = DHEAD * BLOCK_KV_ROWS;

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
    setLoaderSmemPointers(smemQ, smemK, smemV, KTileSize, BLOCK_Q_ROWS, BLOCK_KV_ROWS);
    oneLoaderSetCalculatorAdditionalSmemPointers(smemL, smemM, QTileSize * BLOCK_Q_ROWS, KTileSize * BLOCK_KV_ROWS, BLOCK_Q_ROWS);

    int buf = 0;
    int qIdx = blockIdx.x * BLOCK_Q_ROWS + laneId * ROWS_PER_WARP / WARP;
    float score;
    int kvRow = warp.thread_rank() / (WARP / ROWS_PER_WARP);
    for (int kvtile = BLOCK_KV_ROWS; kvtile < seqLenK; kvtile += BLOCK_KV_ROWS) {
        pipeK.consumer_wait();
        // Each rowGroup handles its calculations
        if (kvtile + kvRow > qIdx) {
            score = -INFINITY;
        } else {
            computeRowNoncausalAttentionScore<ROWS_PER_WARP>(QFrag, smemK[buf], kvRow, laneId, scale, D_HEAD, QFragmentSize, rowGroup);
        }
        // Have only first thread in each subgroup do the softmax updates.
        if (!rowGroup.thread_rank()) {
            rowSoftmax(&smemM, &smemL, qIdx, score);
        }
        rowGroup.sync();

        pipeV.consumer_wait(); // I suspect this line with cause issues.
        /*
        const float* vRowPtr = &smemV[buf][kvRow + rowGroup.thread_rank() * QFragmentSize];
        #pragma unroll
        for (int i = 0; i < QFragmentSize; i++) {
            OFrag[i] = score * vRowPtr[i];
        }

        // Update corresponding section of O
        if (!rowGroup.thread_rank()) {
            float* oRowPtr = &output[qIdx * D_HEAD]
        }*/

        multiplyVStoreO(); // Fix header tomorrow.
        
        buf ^= 1;
    }
}