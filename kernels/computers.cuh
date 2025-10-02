#pragma once

#include "utils.cuh"

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
            // Function for updates when score is infinity (probably more efficient).
        } else {
            float partialDotProduct = 0.0f;
            // QFragmentSize bc is equal due to being along d_head for both.
            const float* kRowPtr = &smemK[buf][kvRow * D_HEAD + laneId * QFragmentSize];
            #pragma unroll
            for (int i = 0; i < QFragmentSize; i++) {
                partialDotProduct += QFrag[i] * kRowPtr[i];
            }
            score = cg::reduce(rowGroup, partialDotProduct, cg::plus<float>()) * scale;

            // Have only first thread in each subgroup do the softmax updates.
            if (!rowGroup.thread_rank()) {
                float newMax = fmaxf(smemM[qIdx], score);
                smemL[qIdx] = (smemL[qIdx] + 1) * expf(smemM[qIdx] - newMax);
                smemM[qIdx] = newMax;
            }
            rowGroup.sync();

            // Multiply against V and accumulate.
            pipeV.consumer_wait();
            const float* vRowPtr = &smemV[buf][kvRow + rowGroup.thread_rank() * QFragmentSize];
            #pragma unroll
            for (int i = 0; i < QFragmentSize; i++) {
                OFrag[i] = score * vRowPtr[i];
            }

            // Update corresponding section of O
            if (!rowGroup.thread_rank()) {
                float* oRowPtr = &output[qIdx * D_HEAD]
            }
        }
        buf ^= 1;
    }
}