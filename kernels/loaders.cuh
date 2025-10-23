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


template<typename scalar_t>
__device__ __forceinline__ void setQVSmemPointers(scalar_t* __restrict__ (&smemQ)[2], scalar_t* __restrict__ (&smemV)[2], int qTileElements, int kvTileElements) {
    extern __shared__ scalar_t smem[];
    smemQ[0] = smem;
    smemQ[1] = smemQ[0] + qTileElements;
    smemV[0] = smem + 2 * qTileElements + 2 * kvTileElements;
    smemV[1] = smemV[0] + kvTileElements;
}


template<typename scalar_t>
__device__ __forceinline__ void setKOSmemPointers(scalar_t* __restrict__ (&smemK)[2], scalar_t* __restrict__ (&smemO), int qTileElements, int kvTileElements) {
    extern __shared__ scalar_t smem[];
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

template<typename scalar_t>
__device__ __forceinline__ void setComputerSmemPointers(scalar_t* __restrict__ (&smemQ)[2], scalar_t* __restrict__ (&smemK)[2], scalar_t* __restrict__ (&smemV)[2], scalar_t* __restrict__ &smemO, int qTileElements, int kvTileElements) {
    setQVSmemPointers(smemQ, smemV, qTileElements, kvTileElements);
    setKOSmemPointers(smemK, smemO, qTileElements, kvTileElements);
}


template<int TILE_SIZE, typename scalar_t>
__device__ __forceinline__ void asyncBufferLoad(const scalar_t* __restrict__ matrix, scalar_t* __restrict__ matrixSmem, int row, int D_HEAD, int laneId, int fragmentSize, pipe_t& pipe) {
    int base = row * D_HEAD + laneId * fragmentSize;
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
}

template<int D_HEAD, typename scalar_t>
__device__ __forceinline__ void asyncWriteO(scalar_t* __restrict__ O, scalar_t* __restrict__ smemO, int absoluteQRow, int laneId, int perThreadFragmentSizeO, pipe_t& pipeO) {
    #pragma unroll
    for (int reads = 0; reads < perThreadFragmentSizeO; reads += 4) {
        int writes = min(perThreadFragmentSizeO - reads, 4);
        if (writes == 4) {
            const float4* smemPtr = reinterpret_cast<const float4*>(smemO + laneId * perThreadFragmentSizeO + reads);
            float4* globalPtr = reinterpret_cast<float4*>(O + absoluteQRow * D_HEAD + laneId * perThreadFragmentSizeO + reads);
            cuda::memcpy_async(globalPtr, smemPtr, sizeof(float4), pipeO);
        } else if (writes == 3) {
            const float2* smemPtr2 = reinterpret_cast<const float2*>(smemO + laneId * perThreadFragmentSizeO + reads);
            float2* globalPtr2 = reinterpret_cast<float2*>(O + absoluteQRow * D_HEAD + laneId * perThreadFragmentSizeO + reads);
            cuda::memcpy_async(globalPtr2, smemPtr2, sizeof(float2), pipeO);

            const float* smemPtr = reinterpret_cast<const float*>(smemO + laneId * perThreadFragmentSizeO + reads + 2);
            float* globalPtr = reinterpret_cast<float*>(O + absoluteQRow * D_HEAD + laneId * perThreadFragmentSizeO + reads + 2);
            cuda::memcpy_async(globalPtr, smemPtr, sizeof(float), pipeO);
        } else if (writes == 2) {
            const float2* smemPtr = reinterpret_cast<const float2*>(smemO + laneId * perThreadFragmentSizeO + reads);
            float2* globalPtr = reinterpret_cast<float2*>(O + absoluteQRow * D_HEAD + laneId * perThreadFragmentSizeO + reads);
            cuda::memcpy_async(globalPtr, smemPtr, sizeof(float2), pipeO);
        } else {
            const float* smemPtr = reinterpret_cast<const float*>(smemO + laneId * perThreadFragmentSizeO + reads);
            float* globalPtr = reinterpret_cast<float*>(O + absoluteQRow * D_HEAD + laneId * perThreadFragmentSizeO + reads);
            cuda::memcpy_async(globalPtr, smemPtr, sizeof(float), pipeO);
        }
    }
}

template<int D_HEAD, int Q_TILE_ROWS, int KV_TILE_ROWS, typename scalar_t>
__device__ __forceinline__ void qvLoaderWarp(
    const scalar_t* __restrict__ Q, const scalar_t* __restrict__ V,
    auto& block, const int& batchSize, const int& numHeads, const int& seqLen,
    auto& pipeQ, auto& pipeK, auto& pipeV, auto& pipeO
) {

    constexpr int qTileElements = D_HEAD * Q_TILE_ROWS;
    constexpr int kvTileElements = D_HEAD * KV_TILE_ROWS;
    constexpr int perThreadfragmentSizeQ = qTileElements / WARP;
    constexpr int perThreadfragmentSizeKV = kvTileElements / WARP;
    uint8_t laneId = threadIdx.x % 32;

    scalar_t* smemQ[2];
    scalar_t* smemV[2];
    setQVSmemPointers<scalar_t>(smemQ, smemV, qTileElements, kvTileElements);

    uint8_t bufQ = 0;
    for (int rowQ = 0; rowQ < batchSize * numHeads * seqLen; rowQ += Q_TILE_ROWS) {
        pipeQ.producer_acquire();
        asyncBufferLoad<qTileElements, scalar_t>(Q, smemQ[bufQ], rowQ, D_HEAD, laneId, perThreadfragmentSizeQ, pipeQ);;
        pipeQ.producer_commit();
        pipeQ.consumer_wait();
        pipeO.producer_acquire();
        uint8_t bufKV = 0;
        for (int rowKV = 0; rowKV < batchSize * numHeads * seqLen; rowKV += KV_TILE_ROWS) {
            pipeK.producer_acquire();
            pipeK.producer_commit();
            pipeK.consumer_wait();
            pipeK.consumer_release();

            pipeV.producer_acquire();
            asyncBufferLoad<kvTileElements, scalar_t>(V, smemV[bufKV], rowKV, D_HEAD, laneId, perThreadfragmentSizeKV, pipeV);
            pipeV.producer_commit();
            pipeV.consumer_wait();
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

template<int D_HEAD, int Q_TILE_ROWS, int KV_TILE_ROWS, typename scalar_t>
__device__ __forceinline__ void koLoaderWarp(
    const scalar_t* __restrict__ K, scalar_t* __restrict__ O,
    auto& block, const int& batchSize, const int& numHeads, const int& seqLen,
    auto& pipeQ, auto& pipeK, auto& pipeV, auto& pipeO
) {
    constexpr int qTileElements = D_HEAD * Q_TILE_ROWS;
    constexpr int kvTileElements = D_HEAD * KV_TILE_ROWS;
    constexpr int perThreadfragmentSizeKV = kvTileElements / WARP;
    constexpr int perThreadfragmentSizeO = qTileElements / WARP; // O is same size as the Q Tile.
    uint8_t laneId = threadIdx.x % 32;

    scalar_t* smemK[2];
    scalar_t* smemO;
    setKOSmemPointers<scalar_t>(smemK, smemO, qTileElements, kvTileElements);

    uint8_t bufQ = 0;
    for (int rowQ = 0; rowQ < batchSize * numHeads * seqLen; rowQ += Q_TILE_ROWS) {
        pipeQ.producer_acquire();
        pipeQ.producer_commit();
        pipeQ.consumer_wait();
        pipeO.producer_acquire();
        uint8_t bufKV = 0;
        for (int rowKV = 0; rowKV < batchSize * numHeads * seqLen; rowKV += KV_TILE_ROWS) {
            pipeK.producer_acquire();
            asyncBufferLoad<kvTileElements, scalar_t>(K, smemK[bufKV], rowKV, D_HEAD, laneId, perThreadfragmentSizeKV, pipeK);
            pipeK.producer_commit();
            pipeK.consumer_wait();
            pipeK.consumer_release();
            
            pipeV.producer_acquire();
            pipeV.producer_commit();
            pipeV.consumer_wait();
            pipeV.consumer_release();
            bufKV ^= 1;
        }
        pipeQ.consumer_release();
        pipeO.producer_commit();
        pipeO.consumer_wait();
        asyncWriteO<D_HEAD, scalar_t>(O, smemO, rowQ, laneId, perThreadfragmentSizeO, pipeO);
        pipeO.consumer_release();
        bufQ ^= 1;
    }
}
