#ifndef UTIL_KERNEL
#define UTIL_KERNEL

#include "Header.cuh"

__host__ __device__ int ceil(const int a, const int b);

__host__ __device__ float* GetRow(float* a, const std::size_t i, const std::size_t pitch);

__host__ __device__ const float* GetRow(const float* a, const std::size_t i, const std::size_t pitch);

__host__ __device__ float* Get(float* a, const std::size_t i, const std::size_t j, const std::size_t pitch);

__host__ __device__ const float* Get(const float* a, const std::size_t i, const std::size_t j, const std::size_t pitch);

__global__ void AdamOptKernel(
    float* param, float* gradient, float* accM, float* accV, const std::size_t t,
    const std::size_t pitchParam, const std::size_t pitchGrad, const std::size_t pitchAccM, const std::size_t pitchAccV,
    const std::size_t feedCount, const std::size_t row, const std::size_t col);

#endif