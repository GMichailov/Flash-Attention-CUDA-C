#include <stdio.h>
#include <cuda_runtime.h>


void check_gpu_props() {
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
    printf("Global memory: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
    printf("Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Shared memory per SM: %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("Registers per block: %d\n", prop.regsPerBlock);
    printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);
    printf("Warp size: %d\n", prop.warpSize);

    int l2Cache = 0;
    cudaDeviceGetAttribute(&l2Cache, cudaDevAttrL2CacheSize, device);
    printf("L2 cache size: %d KB\n", l2Cache / 1024);

    int maxThreadsPerSM = 0;
    cudaDeviceGetAttribute(&maxThreadsPerSM, cudaDevAttrMaxThreadsPerMultiProcessor, device);
    printf("Max threads per SM: %d\n", maxThreadsPerSM);
}

int main() {
    check_gpu_props();
    return 0;
}