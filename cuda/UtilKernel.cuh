#include "Header.h"

__host__ __device__ int ceil(const int a, const int b) {
    return (a + b - 1) / b;
}

__device__ float* GetRow(const float* a, const int i, const std::size_t pitch) {
    return (float*)((char*)a + i * pitch);
}

__device__ float* Get(const float* a, const int i, const int j, const std::size_t pitch) {
    return GetRow(a,i,pitch) + j;
}
