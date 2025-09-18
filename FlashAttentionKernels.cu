#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <cmath>
#include <cuda/pipeline>

#define WARP 32
#define FLOAT_SIZE 4

template<int TILE_SIZE>
__global__ void asyncBufferLoad(const float* __restrict__ matrix, float* __restrict__ matrixSmem, int offset) {
    extern __shared__ float smem[];
    auto pipe = cuda::make_pipeline();
    int thread = threadIdx.x;
    const float4* global_mem = reinterpret_cast<const float4*>(matrix + offset + thread);
    
}
    
template<int DHEAD, int BLOCK_Q_ROWS, int ROWS_PER_WARP>
__global__ void causalFlashAttention(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    const float* __restrict__ L, const float* __restrict__ M,
    int batchSize, int numHeads,
    int seqLenQ, int seqLenK,
    int strideBatchQ, int strideBatchK, int strideBatchV, int strideBatchO,
    int strideHeadQ, int strideHeadK, int strideHeadV, int strideHeadO,
    float scale, 
    int BLOCK_KV_ROWS
)
{
    // Load Q tile
    int warpId = threadIdx.x / WARP;
    int thread = threadIdx.x % 32;
    const float* tile_start = Q + blockIdx.z * strideBatchQ + blockIdx.y * strideHeadQ + blockIdx.x * BLOCK_Q_ROWS;
    constexpr int fragmentSize = DHEAD * BLOCK_Q_ROWS * ROWS_PER_WARP / WARP;
    const float* fragmentStart = tile_start + warpId * ROWS_PER_WARP * DHEAD + thread * fragmentSize;
    float QFrag[fragmentSize];

    // Do reads of float4 and only save what's necessary
    // Logic: Have one thread responsible for float4 and if smaller, calculate how many threads should idle meanwhile.
    if (fragmentSize >= 4 || thread % 4 == 0) {
        #pragma unroll
        for (int reads = 0; reads < fragmentSize; reads+=4) {
            float4 frag = *((const float4*)(fragmentStart + reads));
            int writes = std::min(fragmentSize - reads, 4);
            if (writes > 0) QFrag[reads + 0] = frag.x;
            if (writes > 1) QFrag[reads + 1] = frag.y;
            if (writes > 2) QFrag[reads + 2] = frag.z;
            if (writes > 3) QFrag[reads + 3] = frag.w;
        }
    }
    __syncthreads();    
}