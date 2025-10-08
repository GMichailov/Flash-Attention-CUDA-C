#pragma once

#include "utils.cuh"


__device__ __forceinline__ void oneLoaderSetSmemPointers(float* __restrict__ (&smemQ)[2], float* __restrict__ (&smemK)[2], float* __restrict__ (&smemV)[2], int matrixElements) {
    extern __shared__ float smem[];
    int offset=0;
    smemQ[0] = smem;
    offset += matrixElements;
    smemQ[1] = smem + offset;
    offset += matrixElements;
    smemK[0] = smem + offset;
    offset += matrixElements;
    smemK[1] = smem + offset;
    offset += matrixElements;
    smemV[0] = smem + offset;
    offset += matrixElements;
    smemV[1] = smem + offset;
}


__device__ __forceinline__ void oneLoaderSetCalculatorAdditionalSmemPointers(float* __restrict__ &L, float* __restrict__ &M, int matrixElements, int BLOCK_ROWS) {
    extern __shared__ float smem[];
    int offset=matrixElements * 6;
    L = smem + offset;
    offset += BLOCK_ROWS;
    M = smem + offset;
}


template<int TILE_SIZE>
__device__ __forceinline__ void asyncBufferLoad(const float* __restrict__ matrix, float* __restrict__ matrixSmem, int tileOffset, int laneId, int fragmentSize, pipe_t& pipe) {
    pipe.producer_acquire();
    int base = tileOffset + laneId * fragmentSize;
    #pragma unroll
    for (int reads = 0; reads < fragmentSize; reads += 4) {
        int writes = min(fragmentSize - reads, 4);
        if (writes == 4) {
            const float4* gloablMemPtr = reinterpret_cast<const float4*>(matrix + base + reads);
            float4* smemPtr = reinterpret_cast<float4*>(matrixSmem + laneId * fragmentSize + reads);
            cuda::memcpy_async(smemPtr, gloablMemPtr, sizeof(float4), pipe);
        } else if (writes == 3) {
            const float2* gloablMemPtr2 = reinterpret_cast<const float2*>(matrix + base + reads);
            float2* smemPtr2 = reinterpret_cast<float2*>(matrixSmem + laneId * fragmentSize + reads);
            cuda::memcpy_async(smemPtr2, gloablMemPtr2, sizeof(float2), pipe);

            const float* gloablMemPtr = reinterpret_cast<const float*>(matrix + base + reads + 2);
            float* smemPtr = reinterpret_cast<float*>(matrixSmem + laneId * fragmentSize + reads + 2);
            cuda::memcpy_async(smemPtr, gloablMemPtr, sizeof(float), pipe);
        } else if (writes == 2) {
            const float2* gloablMemPtr = reinterpret_cast<const float2*>(matrix + base + reads);
            float2* smemPtr = reinterpret_cast<float2*>(matrixSmem + laneId * fragmentSize + reads);
            cuda::memcpy_async(smemPtr, gloablMemPtr, sizeof(float2), pipe);
        } else {
            const float* gloablMemPtr = reinterpret_cast<const float*>(matrix + base + reads);
            float* smemPtr = reinterpret_cast<float*>(matrixSmem + laneId * fragmentSize + reads);
            cuda::memcpy_async(smemPtr, gloablMemPtr, sizeof(float), pipe);
        }
    }
    pipe.producer_commit();
}


template<int D_HEAD, int BLOCK_ROWS, int ROWS_PER_WARP>
__device__ __forceinline__ void singleLoaderWarp(
    float* __restrict__ (&smem)[2], 
    pipe_t pipeQ, pipe_t pipeK, pipe_t pipeV, 
    auto block, int laneId,
    int batchSize, int numHeads,
    int seqLenQ, int seqLenK
) {
    constexpr int tileSize = D_HEAD * BLOCK_ROWS;
    constexpr int fragmentSize = D_HEAD / (WARP / ROWS_PER_WARP);

    // Set smem pointers
    float* smemQ[2];
    float* smemK[2];
    float* smemV[2];
    setLoaderSmemPointers(smemQ, smemK, smemV, tileSize);

    // Iteratively load the tiles
    int buf = 0;
    for (int loadingOffsetQ = 0; loadingOffsetQ < seqLenQ * batchSize * numHeads; loadingOffsetQ += BLOCK_ROWS) {
        asyncBufferLoad<QTileSize>(Q, smemQ[buf], loadingOffsetQ, laneId, fragmentSize, pipeQ);
        for(int loadingOffsetKV = 0; loadingOffsetKV < seqLenK * batchSize * numHeads; loadingOffsetKV += BLOCK_ROWS) {
            asyncBufferLoad<KTileSize>(K, smemK[buf], loadingOffsetKV, laneId, fragmentSize, pipeK);
            asyncBufferLoad<KTileSize>(V, smemV[buf], loadingOffsetKV, laneId, fragmentSize, pipeV);
            buf ^= 1;
            // Wait for computation warps to finish calculating on the other buffer.
            __syncthreads();
        }
    }
}