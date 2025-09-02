#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cassert>

inline int calculateSizeBlockQ(cudaDeviceProp& prop, int d_head) {
    // Dynamic Calculation Formula: <= Registers in CTA / (d_head * sizeof(scalar_t)) and Registers in CTA = Registers Total / threads per CTA per SM
    int warpsPerCta = 4;
    int totalRegisters = prop.regsPerMultiprocessor;
    int threadArrRegisters = totalRegisters / (warpsPerCta * prop.warpSize);
    int scalarSize = sizeof(float); // Hardcode as float for now, change later.
    int brMax = threadArrRegisters / (d_head * scalarSize);
    constexpr int maxSize = 128*64;
    int experimentalLimit = maxSize / d_head;
    //return std::min(experimentalLimit, brMax); 
    return 64;
}

inline int calculateSizeBlockKV(cudaDeviceProp& prop, int d_head, int device) {
    // Dynamic Calculation Formula: <= Size L2 Cache / (2 * d_head * sizeof(scalar_t)) [x2 is for double buffering]
    int l2Cache = 0;
    cudaDeviceGetAttribute(&l2Cache, cudaDevAttrL2CacheSize, device);
    int BcMax = l2Cache / (2 * d_head * sizeof(float));
    constexpr int maxSize = 128*64;
    int experimentalLimit = maxSize / d_head;
    //return std::min(experimentalLimit, BcMax);
    return 64;
}


inline int getNumCta(int q_dim, int q_block_size) {
    assert(q_dim % q_block_size == 0 && "Block size of Q rows must cleanly split Q");
    return q_dim / q_block_size;
}