#include "../kernels/FlashAttention.cuh"

int main() {
    twoLoaderMhaFlashAttentionKernel<64, 16, 16><<<1, 256>>>(
        nullptr, nullptr, nullptr, nullptr,
        1, 1, 1, 1.0f, false
    );
    return 0;
}