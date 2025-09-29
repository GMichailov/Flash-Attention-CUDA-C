#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda/pipeline>
#include <cuda_runtime_api.h>

using pipe_t = cuda::pipeline<cuda::thread_scope_block>;


__device__ __forceinline__ void oneLoaderSetSmemPointers(float* __restrict__ (&smemQ)[2], float* __restrict__ (&smemK)[2], float* __restrict__ (&smemV)[2], int qElements, int kvElements) {
    extern __shared__ float smem[];
    int offset=0;
    smemQ[0] = smem;
    offset += qElements;
    smemQ[1] = smem + offset;
    offset += qElements;
    smemK[0] = smem + offset;
    offset += kvElements;
    smemK[1] = smem + offset;
    offset += kvElements;
    smemV[0] = smem + offset;
    offset += kvElements;
    smemV[1] = smem + offset;
}


__device__ __forceinline__ void oneLoaderSetCalculatorAdditionalSmemPointers(float* __restrict__ &L, float* __restrict__ &M, int qElements, int kvElements, int BLOCK_Q_ROWS) {
    extern __shared__ float smem[];
    int offset=qElements * 2 + kvElements*4;
    L = smem + offset;
    offset += BLOCK_Q_ROWS;
    M = smem + offset;
}


template<int TILE_SIZE>
__device__ __forceinline__ void asyncBufferLoad(const float* __restrict__ matrix, float* __restrict__ matrixSmem, int tileOffset, int laneId, int fragmentSize, pipe_t& pipe) {
    if (!laneId) pipe.producer_acquire();
    int base = tileOffset + laneId * fragmentSize;
    #pragma unroll
    for (int reads = 0; reads < fragmentSize; reads += 4) {
        int writes = std::min(fragmentSize - reads, 4);
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
    if (!laneId) pipe.producer_commit();
}

