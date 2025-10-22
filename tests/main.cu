#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

// Include your kernel
#include "../kernels/FlashAttention.cuh"

#define CUDA_CHECK(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

template<int D_HEAD, int Q_TILE_ROWS, int KV_TILE_ROWS>
void test_flash_attention(bool is_causal = false) {
    using namespace std;
    int batchSize = 1;
    int numHeads  = 1;
    int seqLen    = 16;
    float scale   = 1.0f / sqrtf((float)D_HEAD);

    size_t tensorSize = batchSize * numHeads * seqLen * D_HEAD;
    size_t bytes = tensorSize * sizeof(float);

    // Allocate host data
    vector<float> h_Q(tensorSize, 1.0f);
    vector<float> h_K(tensorSize, 1.0f);
    vector<float> h_V(tensorSize, 1.0f);
    vector<float> h_O(tensorSize, 0.0f);

    // Allocate device buffers
    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, bytes));
    CUDA_CHECK(cudaMalloc(&d_K, bytes));
    CUDA_CHECK(cudaMalloc(&d_V, bytes));
    CUDA_CHECK(cudaMalloc(&d_O, bytes));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_O, 0, bytes));

    // Kernel launch configuration
    constexpr int numWarps = Q_TILE_ROWS + 2;
    constexpr int threadsPerBlock = numWarps * WARP;
    dim3 grid(1);
    dim3 block(threadsPerBlock);
    size_t sharedMemBytes = (2 * Q_TILE_ROWS + 4 * KV_TILE_ROWS + Q_TILE_ROWS) * D_HEAD * sizeof(float);

    printf("Launching kernel with %d threads (%d warps)...\n", threadsPerBlock, numWarps);

    // Launch kernel
    twoLoaderMhaFlashAttentionKernel<D_HEAD, Q_TILE_ROWS-2, KV_TILE_ROWS>
        <<<grid, block, sharedMemBytes>>>(d_Q, d_K, d_V, d_O, batchSize, numHeads, seqLen, scale, is_causal);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, bytes, cudaMemcpyDeviceToHost));

    // Print first few outputs
    printf("\nFirst few output values:\n");
    for (int i = 0; i < std::min(16, (int)h_O.size()); ++i)
        printf("%2d: %8.5f\n", i, h_O[i]);

    // Optional: CPU reference sanity check (dot(Q,K)->softmax->V)
    vector<float> ref(tensorSize, 0.0f);
    for (int q = 0; q < seqLen; ++q) {
        vector<float> scores(seqLen);
        for (int k = 0; k < seqLen; ++k) {
            float dot = 0.f;
            for (int d = 0; d < D_HEAD; ++d)
                dot += h_Q[q * D_HEAD + d] * h_K[k * D_HEAD + d];
            if (is_causal && k > q) dot = -1e9f;
            scores[k] = expf(dot * scale);
        }
        float sum = 0.f;
        for (auto s : scores) sum += s;
        for (int k = 0; k < seqLen; ++k) {
            float w = scores[k] / sum;
            for (int d = 0; d < D_HEAD; ++d)
                ref[q * D_HEAD + d] += w * h_V[k * D_HEAD + d];
        }
    }

    float max_diff = 0.f;
    for (int i = 0; i < tensorSize; ++i)
        max_diff = fmaxf(max_diff, fabsf(ref[i] - h_O[i]));

    printf("\nMax absolute difference vs CPU reference: %.6f\n", max_diff);

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
}

int main() {
    printf("Running FlashAttention test (D_HEAD=8, Q_TILE_ROWS=4, KV_TILE_ROWS=4)\n");
    test_flash_attention<16, 4, 4>(false);
    return 0;
}