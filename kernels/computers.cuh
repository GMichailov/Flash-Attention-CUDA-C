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
    auto& block, int batchSize, int numHeads, int seqLen, float scale, bool is_causal
) {
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block> pipeStateQ;
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block> pipeStateK;
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block> pipeStateV;
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block> pipeStateO;
    auto pipeQ = cuda::make_pipeline(block, &pipeStateQ);
    auto pipeK = cuda::make_pipeline(block, &pipeStateK);
    auto pipeV = cuda::make_pipeline(block, &pipeStateV);
    auto pipeO = cuda::make_pipeline(block, &pipeStateO);

    constexpr int qTileElements = D_HEAD * Q_TILE_ROWS;
    constexpr int kvTileElements = D_HEAD * KV_TILE_ROWS;

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
    float score;
    float running_max;
    float running_l;

    // Grab Q and stream KV against it to be able to store O stuff per warp and only keep a Q_TILE_ROWS tile for O.
    for (int globalQRow = 0; globalQRow < batchSize * numHeads * seqLen; globalQRow += Q_TILE_ROWS) {
        score = 0.0f;
        running_max = -INFINITY;
        running_l = 0.0f;
        uint8_t bufKV = 0;
        pipeQ.consumer_wait();
        for (int globalKVRow = 0; globalKVRow < batchSize * numHeads * seqLen; globalKVRow += KV_TILE_ROWS) {
            pipeK.consumer_wait();
            if (is_causal && (globalKVRow + group.meta_group_rank() > globalQRow + warp.meta_group_rank())) score = -INFINITY;
            else computeAttentionScore<Q_TILE_ROWS, KV_TILE_ROWS, D_HEAD>(smemQ[bufQ], smemK[bufKV], scale, warp, group, score);

            float curr_max;
            float curr_l;
            // Each thread then multiplies its fragment against V corresponding V fragment
            // Sum into the first groups fragment for each group in the warp.
            // smemo += this from first group
            group.sync();
            // Sum group to get score and then find max score for warp.
            score = cg::reduce(group, score, cg::plus<float>());
            curr_max = cg::reduce(warp, score, cg::greater<float>());
            curr_max = fmaxf(curr_max, running_max);
            // Sum up scores held by group leaders in warp and find max simultaneously.
            curr_l = (group.thread_rank() == 0) ? expf(score - curr_max) : 0.0f;
            unsigned mask = groupLeaderMask(group.size());
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                curr_l += __shfl_down_sync(mask, curr_l, offset);
            }
            // Broadcast out the currentl to all warp threads.
            curr_l = __shfl_sync(0xFFFFFFFF, curr_l, 0);

            // Multiply against V and Accumulate into O
        }
    }




    //==========================================================================================================================
    // Begin Iterating through tiles.
    /*
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
    */
}