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


__device__ __forceinline__ void setQVSmemPointers(float* __restrict__ (&smemQ)[2], float* __restrict__ (&smemV)[2], int qTileElements, int kvTileElements) {
    extern __shared__ float smem[];
    smemQ[0] = smem;
    smemQ[1] = smemQ[0] + qTileElements;
    smemV[0] = smem + 2 * qTileElements + 2 * kvTileElements;
    smemV[1] = smemV[0] + kvTileElements;
}


__device__ __forceinline__ void setKOSmemPointers(float* __restrict__ (&smemK)[2], float* __restrict__ (&smemO), int qTileElements, int kvTileElements) {
    extern __shared__ float smem[];
    smemK[0] = smem + 2 * qTileElements;
    smemK[1] = smemK[0] + kvTileElements;
    smemO = smem + 2 * qTileElements + 4 * kvTileElements;
}


__device__ __forceinline__ void oneLoaderSetCalculatorAdditionalSmemPointers(float* __restrict__ &L, float* __restrict__ &M, int matrixElements, int BLOCK_ROWS) {
    extern __shared__ float smem[];
    int offset=matrixElements * 6;
    L = smem + offset;
    offset += BLOCK_ROWS;
    M = smem + offset;
}


__device__ __forceinline__ void setComputerSmemPointers(float* __restrict__ (&smemQ)[2], float* __restrict__ (&smemK)[2], float* __restrict__ (&smemV)[2], float* __restrict__ &smemO, int qTileElements, int kvTileElements) {
    setQVSmemPointers(smemQ, smemV, qTileElements, kvTileElements);
    setKOSmemPointers(smemK, smemO, qTileElements, kvTileElements);
}


template<int TILE_SIZE>
__device__ __forceinline__ void asyncBufferLoad(const float* __restrict__ matrix, float* __restrict__ matrixSmem, int tileOffset, int laneId, int fragmentSize, pipe_t& pipe) {
    pipe.producer_acquire();
    int base = tileOffset + laneId * fragmentSize;
    #pragma unroll
    for (int reads = 0; reads < fragmentSize; reads += 4) {
        int writes = min(fragmentSize - reads, 4);
        if (writes == 4) {
            const float4* globalMemPtr = reinterpret_cast<const float4*>(matrix + base + reads);
            float4* smemPtr = reinterpret_cast<float4*>(matrixSmem + laneId * fragmentSize + reads);
            cuda::memcpy_async(smemPtr, globalMemPtr, sizeof(float4), pipe);
        } else if (writes == 3) {
            const float2* globalMemPtr2 = reinterpret_cast<const float2*>(matrix + base + reads);
            float2* smemPtr2 = reinterpret_cast<float2*>(matrixSmem + laneId * fragmentSize + reads);
            cuda::memcpy_async(smemPtr2, globalMemPtr2, sizeof(float2), pipe);

            const float* globalMemPtr = reinterpret_cast<const float*>(matrix + base + reads + 2);
            float* smemPtr = reinterpret_cast<float*>(matrixSmem + laneId * fragmentSize + reads + 2);
            cuda::memcpy_async(smemPtr, globalMemPtr, sizeof(float), pipe);
        } else if (writes == 2) {
            const float2* globalMemPtr = reinterpret_cast<const float2*>(matrix + base + reads);
            float2* smemPtr = reinterpret_cast<float2*>(matrixSmem + laneId * fragmentSize + reads);
            cuda::memcpy_async(smemPtr, globalMemPtr, sizeof(float2), pipe);
        } else {
            const float* globalMemPtr = reinterpret_cast<const float*>(matrix + base + reads);
            float* smemPtr = reinterpret_cast<float*>(matrixSmem + laneId * fragmentSize + reads);
            cuda::memcpy_async(smemPtr, globalMemPtr, sizeof(float), pipe);
        }
    }
    pipe.producer_commit();
}


__device__ __forceinline__ void asyncWriteO(float* __restrict__ O, float* __restrict__ smemO, int absoluteQRow, int laneId, int perThreadFragmentSizeO, pipe_t& pipeO) {
    pipeO.producer_acquire();
    #pragma unroll
    for (int reads = 0; reads < perThreadFragmentSizeO; reads += 4) {
        int writes = min(perThreadFragmentSizeO - reads, 4);
        if (writes == 4) {
            const float4* smemPtr = reinterpret_cast<const float4*>(smemO + laneId * perThreadFragmentSizeO + reads);
            float4* globalPtr = reinterpret_cast<float4*>(O + laneId * perThreadFragmentSizeO + reads);
            cuda::memcpy_async(globalPtr, smemPtr, sizeof(float4), pipeO);
        } else if (writes == 3) {
            const float2* smemPtr2 = reinterpret_cast<const float2*>(smemO + laneId * perThreadFragmentSizeO + reads);
            float2* globalPtr2 = reinterpret_cast<float2*>(O + laneId * perThreadFragmentSizeO + reads);
            cuda::memcpy_async(globalPtr2, smemPtr2, sizeof(float2), pipeO);

            const float* smemPtr = reinterpret_cast<const float*>(smemO + laneId * perThreadFragmentSizeO + reads + 2);
            float* globalPtr = reinterpret_cast<float*>(O + laneId * perThreadFragmentSizeO + reads + 2);
            cuda::memcpy_async(globalPtr, smemPtr, sizeof(float), pipeO);
        } else if (writes == 2) {
            const float2* smemPtr = reinterpret_cast<const float2*>(smemO + laneId * perThreadFragmentSizeO + reads);
            float2* globalPtr = reinterpret_cast<float2*>(O + laneId * perThreadFragmentSizeO + reads);
            cuda::memcpy_async(globalPtr, smemPtr, sizeof(float2), pipeO);
        } else {
            const float* smemPtr = reinterpret_cast<const float*>(smemO + laneId * perThreadFragmentSizeO + reads);
            float* globalPtr = reinterpret_cast<float*>(O + laneId * perThreadFragmentSizeO + reads);
            cuda::memcpy_async(globalPtr, smemPtr, sizeof(float), pipeO);
        }
    }
    pipeO.producer_commit();
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

template<int D_HEAD, int Q_TILE_ROWS, int KV_TILE_ROWS>
__device__ __forceinline__ void qvLoaderWarp(
    float* __restrict__ Q, float* __restrict__ V,
    auto& block, const int& batchSize, const int& numHeads, const int& seqLen
) {
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block> pipeStateQ;
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block> pipeStateV;
    auto pipeQ = cuda::make_pipeline(block, &pipeStateQ);
    auto pipeV = cuda::make_pipeline(block, &pipeStateV);

    constexpr int qTileElements = D_HEAD * Q_TILE_ROWS;
    constexpr int kvTileElements = D_HEAD * KV_TILE_ROWS;
    constexpr int perThreadfragmentSizeQ = qTileElements / WARP;
    constexpr int perThreadfragmentSizeKV = kvTileElements / WARP;
    uint8_t laneId = threadIdx.x % 32;

    float* smemQ[2];
    float* smemV[2];
    setQVSmemPointers(smemQ, smemV, qTileElements, kvTileElements);

    uint8_t bufQ = 0;
    for (int rowQ = 0; rowQ < batchSize * numHeads * seqLen; rowQ += Q_TILE_ROWS) {
        asyncBufferLoad<qTileElements>(Q, smemQ[bufQ], rowQ, laneId, perThreadfragmentSizeQ, pipeQ);
        uint8_t bufKV = 0;
        for (int rowKV = 0; rowKV < batchSize * numHeads * seqLen; rowKV += KV_TILE_ROWS) {
            asyncBufferLoad<kvTileElements>(V, smemV[bufKV], rowKV, laneId, perThreadfragmentSizeKV, pipeV);
            bufKV ^= 1;
            __syncthreads();
        }
        bufQ ^= 1;
    }
}

template<int D_HEAD, int Q_TILE_ROWS, int KV_TILE_ROWS>
__device__ __forceinline__ void koLoaderWarp(
    float* __restrict__ K, float* __restrict__ O,
    auto& block, const int& batchSize, const int& numHeads, const int& seqLen
) {
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block> pipeStateK;
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block> pipeStateO;
    auto pipeK = cuda::make_pipeline(block, &pipeStateK);
    auto pipeO = cuda::make_pipeline(block, &pipeStateO);

    constexpr int qTileElements = D_HEAD * Q_TILE_ROWS;
    constexpr int kvTileElements = D_HEAD * KV_TILE_ROWS;
    constexpr int perThreadfragmentSizeKV = kvTileElements / WARP;
    constexpr int perThreadfragmentSizeO = qTileElements / WARP; // O is same size as the Q Tile.
    uint8_t laneId = threadIdx.x % 32;

    float* smemK[2];
    float* smemO[2];
    setKOSmemPointers(smemK, smemO, qTileElements, kvTileElements);

    uint8_t bufQ = 0;
    for (int rowQ = 0; rowQ < batchSize * numHeads * seqLen; rowQ += Q_TILE_ROWS) {
        for (int rowKV = 0; rowKV < batchSize * numHeads * seqLen; rowKV += KV_TILE_ROWS) {
            asyncBufferLoad<kvTileElements>(K, smemK[bufQ], rowKV, laneId, perThreadfragmentSizeKV, pipeK);
            __syncthreads();
        }
        asyncWriteO(O, smemO, rowQ, laneId, perThreadfragmentSizeO, pipeO);
        bufQ ^= 1;
    }
}