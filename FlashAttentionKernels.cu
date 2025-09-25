#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <cmath>
#include <cuda/pipeline>

#define WARP 32
#define FLOAT_SIZE 4

using pipe_t = cuda::pipeline<cuda::thread_scope_block>;

__shared__ pipe_t pipeK;
__shared__ pipe_t pipeV;


__device__ __forceinline__ void setSmemPointers(
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
    extern __shared__ float smem[];
    int warpId = threadIdx.x / WARP;
    int thread = threadIdx.x % WARP;

    // Instantiate pipes. (Could I reuse the same pipes throughout rather than per attention block?)
    if (thread == 0) {
        new (&pipeK) pipe_t();
    } else if (thread == 1) {
        new (&pipeV) pipe_t();
    }
    __syncthreads();

    constexpr int QTileSize = DHEAD * BLOCK_Q_ROWS;
    constexpr int QFragmentSize = QTileSize / WARP;
    constexpr int KTileSize = DHEAD * BLOCK_KV_ROWS;
    constexpr int KFragmentSize = KTileSize / WARP;

    float* smemK[2];
    float* smemV[2];
    float* smemO;
    float* smemL;
    float* smemM;
    float QFrag[QFragmentSize];

    setSmemPointers(smemK, smemV, smemO, smemL, smemM, KTileSize, QTileSize, BLOCK_Q_ROWS);
    
    // Async load first KV tiles with warps 0 and 1.
    if (warpId == 0) {
        // Load K
        asyncBufferLoad<KTileSize>(
            K, smemK[0],
            0, thread, KFragmentSize, pipeK
        );
    } else if (warpId == 1) {
        // Load V
        asyncBufferLoad<KTileSize>(
            V, smemV[0],
            0, thread, KFragmentSize, pipeV
        );
    } else {
        // Load Q tile
        loadQRegisters<DHEAD, BLOCK_Q_ROWS, ROWS_PER_WARP>(
            Q, QFrag,
            blockIdx.z * strideBatchQ, blockIdx.y * strideHeadQ,
            warpId, thread, QFragmentSize
        );
    }
    __syncthreads();
    int buf = 0;
    for (int kvtile = BLOCK_KV_ROWS; kvtile < seqLenK; kvtile += BLOCK_KV_ROWS) {
        int nextBuf ^= 1;

        // Call async loading first.
        if (kvtile < numTiles - 1) {
            if (warpId == 0) asyncBufferLoad<KTileSize>(K, smemK[nextBuf], (kvtile+1)*BLOCK_KV_ROWS, thread, KFragmentSize, pipeK);
            if (warpId == 1) asyncBufferLoad<KTileSize>(V, smemV[nextBuf], (kvtile+1)*BLOCK_KV_ROWS, thread, KFragmentSize, pipeV);
        }
        
        if (warpId >= 2) {
            pipeK.consumer_wait();
            // Mask: 32 bit int where each bit represents which threads are broadcasted to in the warp.
            // mask is created by creating number of consecutive ones at back for threads listening and then pushes it back however many times as necessary.
            unsigned mask = ((1u << WARP / ROWS_PER_WARP) - 1) << (((thread * ROWS_PER_WARP) / WARP) * WARP / ROWS_PER_WARP);
            for (int kvRow = 0; kvRow < BLOCK_KV_ROWS; ++kvRow) {
                const float* kRowPtr = &smemK[buf][kvRow * D_HEAD + thread * QFragmentSize]; // QFragmentSize bc is equal due to being along d_head.
                float partialDotProduct = 0.0f;

                #pragma unroll
                for (int i = 0; i < QFragmentSize; i++) {
                    partialDotProduct += QFrag[i] * kRowPtr[i];
                }

                
            }
        }
    }
}