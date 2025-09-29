#include "FlashAttention.cuh"

// Kernel that uses only 1 warp for loading from HBM into SRAM (High compute)
template<int DHEAD, int BLOCK_Q_ROWS, int BLOCK_KV_ROWS, int ROWS_PER_WARP>
__global__ void oneLoaderMhaFlashAttentionKernel(
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