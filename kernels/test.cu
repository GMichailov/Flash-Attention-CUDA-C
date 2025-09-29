#include "loaders.cuh"

#include "loaders.cuh"
#include <cstdio>

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
    const int qElements = 256;      // e.g., BLOCK_Q_ROWS * D_HEAD = 32 * 8
    const int kvElements = 512;     // e.g., BLOCK_KV * D_HEAD = 64 * 8
    const int BLOCK_Q_ROWS = 32;

    // Total shared memory size = Q(2) + K(2) + V(2) + L + M
    int totalSmemFloats = 2 * qElements + 4 * kvElements + 2 * BLOCK_Q_ROWS;
    size_t smemBytes = totalSmemFloats * sizeof(float);

    printf("Launching with %zu bytes of shared memory\n", smemBytes);

    testSmemLayout<<<1, 1, smemBytes>>>(qElements, kvElements, BLOCK_Q_ROWS);
    cudaDeviceSynchronize();
}

int main() {
    test_loaders();
    return 0;
}