#ifndef UTIL_KERNEL
#define UTIL_KERNEL

#include "Header.cuh"

__host__ __device__ int ceil(const int a, const int b);

template<typename T>
__host__ __device__ T* GetRow(T* a, const std::size_t i, const std::size_t pitch) {
    return (T*)((char*)a + i * pitch);
}

template<typename T>
__host__ __device__ T* Get(T* a, const std::size_t i, const std::size_t j, const std::size_t pitch) {
    return GetRow(a, i, pitch) + j;
}

__global__ void AdamOptKernel(
    float* param, float* gradient, float* accM, float* accV, std::size_t* t,
    const std::size_t pitchParam, const std::size_t pitchGrad, const std::size_t pitchAccM, const std::size_t pitchAccV,
    const std::size_t row, const std::size_t col);

#endif