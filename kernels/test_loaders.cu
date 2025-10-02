#include "loaders.cuh"
#include <cstdio>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void testSmemLayout(int qElements, int kvElements, int BLOCK_Q_ROWS) {
    // Declare arrays for smem pointer storage
    float* smemQ[2];
    float* smemK[2];
    float* smemV[2];
    float* L;
    float* M;

    // Call the loader functions (this is what you’re testing)
    oneLoaderSetSmemPointers(smemQ, smemK, smemV, qElements, kvElements);
    oneLoaderSetCalculatorAdditionalSmemPointers(L, M, qElements, kvElements, BLOCK_Q_ROWS);

    // TEMP: print out pointer offsets relative to smem[0]
    extern __shared__ float smem[];
    printf("Q0 offset: %ld\n", smemQ[0] - smem);
    printf("Q1 offset: %ld\n", smemQ[1] - smem);
    printf("K0 offset: %ld\n", smemK[0] - smem);
    printf("K1 offset: %ld\n", smemK[1] - smem);
    printf("V0 offset: %ld\n", smemV[0] - smem);
    printf("V1 offset: %ld\n", smemV[1] - smem);
    printf("L  offset: %ld\n", L - smem);
    printf("M  offset: %ld\n", M - smem);
}


void test_loaders() {
    const int qElements = 256;
    const int kvElements = 512;
    const int BLOCK_Q_ROWS = 32;

    // Total shared memory size = Q(2) + K(2) + V(2) + L + M
    int totalSmemFloats = 2 * qElements + 4 * kvElements + 2 * BLOCK_Q_ROWS;
    size_t smemBytes = totalSmemFloats * sizeof(float);

    printf("Launching with %zu bytes of shared memory\n", smemBytes);

    testSmemLayout<<<1, 1, smemBytes>>>(qElements, kvElements, BLOCK_Q_ROWS);
    cudaDeviceSynchronize();
}

template<int TILE_SIZE>
__global__ void testAsyncBufferLoad(const float* __restrict__ matrix,
                                    float* __restrict__ out,
                                    int fragmentSize) {
    extern __shared__ float smem[];
    cg::thread_block block = cg::this_thread_block();
    constexpr int stages = 1;

    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, stages> shared_state;
    auto pipeline = cuda::make_pipeline(block, &shared_state);

    int laneId = threadIdx.x;   // each thread gets its own lane
    int tileOffset = 0;

    // All lanes issue async copies into shared memory
    asyncBufferLoad<TILE_SIZE>(matrix, smem, tileOffset, laneId, fragmentSize, pipeline);

    // Wait for all copies to complete
    pipeline.consumer_wait();

    // Write results back to global for verification
    for (int i = 0; i < fragmentSize; i++) {
        out[laneId * fragmentSize + i] = smem[laneId * fragmentSize + i];
    }
}

void test_async_loader() {
    const int TILE_SIZE = 128;
    const int NUM_THREADS = 8;
    const int FRAGMENT_SIZE = 8;

    int total = NUM_THREADS * FRAGMENT_SIZE;
    size_t bytes = total * sizeof(float);

    // Host data
    float* h_in = new float[total];
    float* h_out = new float[total];
    for (int i = 0; i < total; i++) h_in[i] = (float)(i + 1);

    // Device data
    float* d_in;
    float* d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    size_t smemBytes = total * sizeof(float);
    testAsyncBufferLoad<TILE_SIZE><<<1, NUM_THREADS, smemBytes>>>(d_in, d_out, FRAGMENT_SIZE);
    cudaDeviceSynchronize();

    // Copy back and verify
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    printf("Verifying asyncBufferLoad results:\n");
    for (int i = 0; i < total; i++) {
        printf("h_out[%d] = %f (expected %f)\n", i, h_out[i], h_in[i]);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;
}

int main() {
    test_async_loader();
    return 0;
}