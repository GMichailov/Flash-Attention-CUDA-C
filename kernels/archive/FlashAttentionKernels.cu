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

__shared__ pipe_t pipeQ;
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


__device__ __forceinline__ void setCalculatorSmemPointers(float* __restrict__ &O, float* __restrict__ &L, float* __restrict__ &M, int kvElements, int qElements, int BLOCK_Q_ROWS) {
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


// Kernel that uses only 1 warp for loading from HBM into SRAM (High compute)
template<int DHEAD, int BLOCK_Q_ROWS, int BLOCK_KV_ROWS, int ROWS_PER_WARP>
__global__ void causalFlashAttention2(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    const float* __restrict__ L, const float* __restrict__ M,
    int batchSize, int numHeads,
    int seqLenQ, int seqLenK,
    int strideBatchQ, int strideBatchK, int strideBatchV, int strideBatchO,
    int strideHeadQ, int strideHeadK, int strideHeadV, int strideHeadO,
    float scale, 
    int BLOCK_KV_ROWS, bool is_causal
)
{
    extern __shared__ float smem[];
    int warpId = threadIdx.x / WARP;
    int laneId = threadIdx.x % WARP;
    // Split off loader warps
    if (warpId == 0) {
        constexpr int QTileSize = DHEAD * BLOCK_Q_ROWS;
        constexpr int QFragmentSize = QTileSize / WARP;
        constexpr int KTileSize = DHEAD * BLOCK_KV_ROWS;
        constexpr int KFragmentSize = KTileSize / WARP;

        // Instantiate pipes
        if (laneId == 0) {
            new (&pipeQ) pipe_t();
        } else if (laneId == 1) {
            new (&pipeK) pipe_t();
        } else if (laneId == 2) {
            new (&pipeV) pipe_t();
        }
        // Set smem pointers
        float* smemQ[2];
        float* smemK[2];
        float* smemV[2];
        setLoaderSmemPointers(smemQ, smemK, smemV, KTileSize, BLOCK_Q_ROWS, BLOCK_KV_ROWS);

        // Iteratively load the tiles
        int buf = 0;
        
        for (int loadingOffsetQ = 0; loadingOffsetQ < seqLenQ; loadingOffsetQ += BLOCK_Q_ROWS) {
            asyncBufferLoad<QTileSize>(Q, smemQ[buf], loadingOffset, laneId, QFragmentSize, pipeQ);
            for(int loadingOffsetKV = 0; loadingOffsetKV < seqLenK; loadingOffsetKV += BLOCK_KV_ROWS) {
                asyncBufferLoad<KTileSize>(K, smemK[buf], loadingOffsetKV, laneId, KFragmentSize, pipeK);
                asyncBufferLoad<KTileSize>(V, smemV[buf], loadingOffsetKV, laneId, KFragmentSize, pipeV);
                buf ^= 1;
                __syncthreads();
            }
        }
    } else {
        constexpr int QTileSize = DHEAD * BLOCK_Q_ROWS;
        constexpr int QFragmentSize = QTileSize / WARP;
        constexpr int KTileSize = DHEAD * BLOCK_KV_ROWS;

        // Create partitions per Q row
        auto warp = cg::tiled_partition<WARP>(cg::this_thread_block());
        auto rowGroup = cg::tiled_partition<WARP / ROWS_PER_WARP>(warp);

        // Set smem pointers needed for calculations
        float* smemQ[2];
        float* smemK[2];
        float* smemV[2];
        float* smemL;
        float* smemM;
        float QFrag[QFragmentSize];
        float OFrag[QFragmentSize];
        setCalculatorSmemPointers(smemQ, smemK, smemV, smemL, smemM, KTileSize, QTileSize, BLOCK_Q_ROWS);

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
}

// Kernel that uses 2 warps for loading from HBM into SRAM 
// Warp 0 does Q and then V.
// Warp 1 does K and then O

// Kernel that uses 3 warps for loading from HBM into SRAM
// Warps 0-2 do Q and O, K, V respectively.