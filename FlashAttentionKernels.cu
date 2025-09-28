#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <cmath>
#include <cuda/pipeline>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define WARP 32
#define FLOAT_SIZE 4

using pipe_t = cuda::pipeline<cuda::thread_scope_block>;

__shared__ pipe_t pipeK;
__shared__ pipe_t pipeV;


__device__ __forceinline__ void setLoaderSmemPointers(float* __restrict__ (&smemK)[2], float* __restrict__ (&smemV)[2], int kvElements) {
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


__device__ __forceinline__ void setCalculatorSmemPointers(
    float* __restrict__ (&smemK)[2], float* __restrict__ (&smemV)[2],
    float* __restrict__ &O, float* __restrict__ &L, float* __restrict__ &M,
    int kvElements, int qElements, int BLOCK_Q_ROWS
)
{
    extern __shared__ float smem[];
    int offset=0;
    smemK[0] = smem + offset;
    offset += kvElements;
    smemK[1] = smem + offset;
    offset += kvElements;
    smemV[0] = smem + offset;
    offset += kvElements;
    smemV[1] = smem + offset;
    offset += kvElements;
    O = smem + offset;
    offset += qElements;
    L = smem + offset;
    offset += BLOCK_Q_ROWS;
    M = smem + offset;
}


template<int DHEAD, int BLOCK_Q_ROWS, int ROWS_PER_WARP>
__device__ __forceinline__ void loadQRegisters(
    const float* __restrict__ Q,
    float* __restrict__ QFrag,
    int batch, int head, int warpId, int thread, int fragmentSize
) {
    const float* tile_start = Q + blockIdx.z * strideBatchQ + blockIdx.y * strideHeadQ + blockIdx.x * BLOCK_Q_ROWS;
    const float* fragmentStart = tile_start + warpId * ROWS_PER_WARP * DHEAD + thread * fragmentSize;

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
}

template<int TILE_SIZE>
__device__ __forceinline__ void asyncBufferLoad(
    const float* __restrict__ matrix, float* __restrict__ matrixSmem,
    int tileOffset, int thread, int fragmentSize,
    pipe_t& pipe
) {
    if (thread == 0) pipe.producer_acquire();
    // Change this to iterate through the chunks
    int base = tileOffset + thread * fragmentSize;
    #pragma unroll
    for (int reads = 0; reads < fragmentSize / 4; ++reads) {
        if (thread * fragmentSize <= TILE_SIZE) {
            const float4* globalMemPtr = reinterpret_cast<const float4*>(matrix + base + reads * 4);
            float4* smemPtr = reinterpret_cast<float4*>(matrixSmem + base + reads * 4);
            cuda::memcpy_async(pipe, smemPtr, globalMemPtr, sizeof(float4));
        }
    }
    if (!thread) pipe.producer_commit();
}


template<int DHEAD, int BLOCK_Q_ROWS, int BLOCK_KV_ROWS, int ROWS_PER_WARP>
__global__ void causalFlashAttention2(
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
    extern __shared__ float smem[];
    int warpId = threadIdx.x / WARP;
    int laneId = threadIdx.x % WARP;
    // Split off loader warps
    if (warpId == 0 || warpId == 1) {
        constexpr int KTileSize = DHEAD * BLOCK_KV_ROWS;
        constexpr int KFragmentSize = KTileSize / WARP;

        // Instantiate pipes
        if (laneId == 0) {
            new (&pipeK) pipe_t();
        } else if (laneId == 1) {
            new (&pipeV) pipe_t();
        }
        // Set smem pointers
        float* smemK[2];
        float* smemV[2];
        setLoaderSmemPointers(smemK, smemV, KTileSize, BLOCK_KV_ROWS);
        // Preload first K and V tiles while Q is being loaded into registers
        if (warpId == 0) {
            asyncBufferLoad<KTileSize>(K, smemK[0], 0, laneId, KFragmentSize, pipeK);
        } else {
            asyncBufferLoad<KTileSize>(V, smemV[0], 0, laneId, KFragmentSize, pipeV);
        }
        // Iteratively load the tiles
        int buf = 0;
        for(int loadingOffset = BLOCK_KV_ROWS; loadingOffset < seqLenK; loadingOffset += BLOCK_KV_ROWS) {
            buf ^= 1;
            if(warpId == 0) asyncBufferLoad<KTileSize>(K, smemK[buf], loadingOffset, laneId, KFragmentSize, pipeK);
            else asyncBufferLoad<KTileSize>(V, smemV[buf], loadingOffset, laneId, KFragmentSize, pipeV);
            __syncthreads();
        }
    } else {
        constexpr int QTileSize = DHEAD * BLOCK_Q_ROWS;
        constexpr int QFragmentSize = QTileSize / WARP;
        constexpr int KTileSize = DHEAD * BLOCK_KV_ROWS;

        // Create partitions per Q row
        auto warp = cg::tiled_partition<WARP>(cg::this_thread_block());
        auto rowGroup = cg::tiled_partition<WARP / ROWS_PER_WARP>(warp);

        // Set smem pointers needed for calculations
        float* smemK[2];
        float* smemV[2];
        float* smemO;
        float* smemL;
        float* smemM;
        float QFrag[QFragmentSize];
        setCalculatorSmemPointers(smemK, smemV, smemO, smemL, smemM, KTileSize, QTileSize, BLOCK_Q_ROWS);

        // Load Q into registers
        loadQRegisters<DHEAD, BLOCK_Q_ROWS, ROWS_PER_WARP>(Q, QFrag, blockIdx.z * strideBatchQ, blockIdx.y * strideHeadQ, warpId, laneId, QFragmentSize);

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
            }
            buf ^= 1;
        }
    }
}