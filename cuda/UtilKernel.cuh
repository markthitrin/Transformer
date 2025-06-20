#ifndef UTIL_KERNEL
#define UTIL_KERNEL

#include "Header.cuh"

__host__ __device__ int ceil(const int a, const int b) {
    return (a + b - 1) / b;
}

__device__ float* GetRow(const float* a, const int i, const std::size_t pitch) {
    return (float*)((char*)a + i * pitch);
}

__device__ float* Get(const float* a, const int i, const int j, const std::size_t pitch) {
    return GetRow(a,i,pitch) + j;
}

__global__ void AdamOptKernel(
    const float* param, const float* gradient, const float* accM, const float* accV, const float t,
    const std::size_t pitchParam, const std::size_t pitchGrad, const std::size_t pitchAccM, const std::size_t pitchAccV,
    const std::size_t row, const std::size_t col) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    const float invPowBeta1 = __frcp_rn(1.0f - std::pow(beta1, t));
    const float invPowBeta2 = __frcp_rn(1.0f - std::pow(beta2, t));
    if(r < row && c < col) {
        *Get(accM, r, c, pitchAccM) = *Get(accM, r, c, pitchAccM) * beta1 + *Get(gradient, r, c, pitchGrad) * (1.0f - beta1);
        *Get(accV, r, c, pitchAccM) = *Get(accV, r, c, pitchAccM) * beta2 + *Get(gradient, r, c, pitchGrad) * *Get(gradient, r, c, pitchGrad) * (1.0f - beta2);
        float mHat = *Get(accM, r, c, pitchAccM) * invPowBeta1;
        float vHat = *Get(accV, r, c, pitchAccM) * invPowBeta2;
        *Get(param, r, c, pitchParam) -= lr * mHat / (std::sqrt(vHat) + eps);
    }
}

#endif