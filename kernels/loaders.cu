#include "loaders.cuh"

using pipe_t = cuda::pipeline<cuda::thread_scope_block>;


__device__ __forceinline__ void setOneLoaderSmemPointers(float* __restrict__ (&smemK)[2], float* __restrict__ (&smemV)[2], int kvElements) {
    extern __shared__ float smem[];
    int offset=0;
    smemK[0] = smem + offset;
    offset += kvElements;
    smemK[1] = smem + offset;
    offset += kvElements;
    smemV[0] = smem + offset;
    offset += kvElements;
    smemV[1] = smem + offset;
}


__device__ __forceinline__ void setOneCalculatorSmemPointers(float* __restrict__ &O, float* __restrict__ &L, float* __restrict__ &M, int kvElements, int qElements, int BLOCK_Q_ROWS) {
    extern __shared__ float smem[];
    int offset=kvElements*4;
    O = smem + offset;
    offset += qElements;
    L = smem + offset;
    offset += BLOCK_Q_ROWS;
    M = smem + offset;
}


template<int DHEAD, int BLOCK_Q_ROWS, int ROWS_PER_WARP>
__device__ __forceinline__ void loadQRegisters(const float* __restrict__ Q, float* __restrict__ QFrag, int batch, int head, int warpId, auto laneId, int fragmentSize) {
    // tile_start = Q + blockIdx.z * strideBatchQ + blockIdx.y * strideHeadQ + blockIdx.x * BLOCK_Q_ROWS;
    const float* fragmentStart = Q + blockIdx.z * strideBatchQ + blockIdx.y * strideHeadQ + blockIdx.x * BLOCK_Q_ROWS + warpId * ROWS_PER_WARP * DHEAD + laneId * fragmentSize;
    // Do reads of float4 if possible and only save what's necessary
    #pragma unroll
    for (int reads = 0; reads < fragmentSize; reads += 4) {
        int writes = std::mid(fragmentSize - reads, 4);
        if (writes == 4) {
            float4 frag = *((const float4*)(fragmentStart + reads));
            QFrag[reads] = frag.x;
            QFrag[reads + 1] = frag.y;
            QFrag[reads + 2] = frag.z;
            QFrag[reads + 3] = frag.w;
        } else if (writes == 3) {
            // Not 8 or 16 bit aligned so have to do with reads of float and float2
            float2 frag = *((const float2*)(fragmentStart + reads));
            QFrag[reads] = frag.x;
            QFrag[reads + 1] = frag.y;
            QFrag[reads + 2] = *(fragmentStart + reads + 2);
        } else if (writes == 2) {
            float2 frag = *((const float2*)(fragmentStart + reads));
            QFrag[reads] = frag.x;
            QFrag[reads + 1] = frag.y;
        } else {
            QFrag[reads] = *(fragmentStart + reads);
        }
    }
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
            cuda::memcpy_async(pipe, smemPtr, gloablMemPtr, sizeof(float4));
        } else if (writes == 3) {
            const float2* gloablMemPtr2 = reinterpret_cast<const float2*>(matrix + base + reads);
            float2* smemPtr2 = reinterpret_cast<float2*>(matrixSmem + laneId * fragmentSize + reads);
            cuda::memcpy_async(pipe, smemPtr2, gloablMemPtr2, sizeof(float2));

            const float gloablMemPtr = reinterpret_cast<const float*>(matrix + base + reads + sizeof(float2));
            float* smemPtr = reinterpret_cast<float*>(matrixSmem + laneId * fragmentSize + reads + sizeof(float2));
            cuda::memcpy_async(pipe, smemPtr, gloablMemPtr, sizeof(float));
        } else if (writes == 2) {
            const float2* gloablMemPtr = reinterpret_cast<const float2*>(matrix + base + reads);
            float2* smemPtr = reinterpret_cast<float2*>(matrixSmem + laneId * fragmentSize + reads);
            cuda::memcpy_async(pipe, smemPtr, gloablMemPtr, sizeof(float2));
        } else {
            const float gloablMemPtr = reinterpret_cast<const float*>(matrix + base + reads);
            float* smemPtr = reinterpret_cast<float*>(matrixSmem + laneId * fragmentSize + reads);
            cuda::memcpy_async(pipe, smemPtr, gloablMemPtr, sizeof(float));
        }
    }
    if (!laneId) pipe.producer_commit();
}


