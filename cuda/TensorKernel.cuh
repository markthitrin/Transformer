#ifndef TENSOR_KERNEL
#define TENSOR_KERNEL

#include "Header.cuh"
#include "Tensor.cuh"

__global__ void PlusKernel(
    const float* A, const float* B, float* C,
    const std::size_t pitchA, const std::size_t pitchB, const std::size_t pitchC,
    const std::size_t row, const std::size_t col);
__global__ void PlusKernel(
    const float* A, const float x, float* C,
    const std::size_t pitchA, const std::size_t pitchC,
    const std::size_t row, const std::size_t col);
__global__ void PlusInplaceKernel(
    float* A, const float* B,
    const std::size_t pitchA, const std::size_t pitchB,
    const std::size_t row, const std::size_t col);
__global__ void PlusBatchKernel(
    const float* A, const float* B, float* C,
    const std::size_t pitchA,const std::size_t pitchB, const std::size_t pitchC,
    const std::size_t batch, const std::size_t row, const std::size_t col);
__global__ void PlusInplaceBatchKernel(
    float* A, const float* B,
    const std::size_t pitchA, const std::size_t pitchB,
    const std::size_t batch, const std::size_t row, const std::size_t col);
__global__ void PlusReduceInplaceBatchKernel(
    float* A, const float* B,
    const std::size_t pitchA, const std::size_t pitchB,
    const std::size_t batch, const std::size_t row, const std::size_t col);
__global__ void PlusProductReduceInplaceBatchKernel(
    float* A, const float* B, const float* C, 
    const std::size_t pitchA, const std::size_t pitchB, const std::size_t pitchC,
    const std::size_t batch, const std::size_t row, const std::size_t col);

__global__ void SubKernel(
    const float* A, const float* B, float* C,
    const std::size_t pitchA, const std::size_t pitchB, const std::size_t pitchC,
    const std::size_t row, const std::size_t col);
__global__ void SubKernel(
    const float* A, const float x, float* C,
    const std::size_t pitchA, const std::size_t pitchC,
    const std::size_t row, const std::size_t col);

__global__ void MulKernel(
    const float* A, const float* B, float* C,
    const std::size_t pitchA, const std::size_t pitchB, const std::size_t pitchC,
    const std::size_t row, const std::size_t col);
__global__ void MulKernel(
    const float* A, const float x, float* C,
    const std::size_t pitchA, const std::size_t pitchC,
    const std::size_t row, const std::size_t col);
__global__ void MulInplaceKernel(
    float* A, const float x,
    const std::size_t pitchA,
    const std::size_t row, const std::size_t col);

__global__ void DivKernel(
    const float* A, const float* B, float* C,
    const std::size_t pitchA, const std::size_t pitchB, const std::size_t pitchC,
    const std::size_t row, const std::size_t col);

__global__ void SetKernel(
    float* A, const float x,
    const std::size_t pitchA,
    const std::size_t row, const std::size_t col);

__global__ void ReduceSumOfProductKernel(
	const float* A, const float* B, float* C,
	const std::size_t pitchA, const std::size_t pitchB,
	const std::size_t row, const std::size_t col);
__global__ void ReduceSumKernel(
	const float* A, float* C,
	const std::size_t pitchA,
	const std::size_t row, const std::size_t col);

__global__ void MeanKernel(
	const float* A, float* C,
	const std::size_t pitchA,
	const std::size_t row, const std::size_t col);
__global__ void StdKernel(
	const float* A, const float* mean, float* C,
	const std::size_t pitchA,
	const std::size_t row, const std::size_t col);

__global__ void ReduceMaxKernel(
    const float* A, const float* C,
    const std::size_t pitchA, const std::size_t pitchC,
    const std::size_t row, const std::size_t col);

__global__ void ApplyLookAheadMaskBatchKernel(
    float* A, const int seq, const float x,
    const std::size_t pitchA, 
    const std::size_t batch, const std::size_t row, const std::size_t col);
__global__ void ApplyPaddingMaskBatchKernel(
    float* A, const int seq, const float x,
    const std::size_t pitchA, 
    const std::size_t batch, const std::size_t row, const std::size_t col);
__global__ void ApplyCrossPaddingMaskBatchKernel(
    float* A, const int seq, const float x,
    const std::size_t pitchA, 
    const std::size_t batch, const std::size_t row, const std::size_t col);

// Need -use_fast_math
__global__ void GetPositionalEncodeKernel(
    float* A, const std::size_t pitchA, 
    const std::size_t row, const std::size_t col);

// A : d1 * d2
// B : d2 * d3
// C : d1 * d3
__global__ void MatMulKernelAB(
    const float* A, const float* B, float* C,
    const std::size_t pitchA, const std::size_t pitchB, const std::size_t pitchC,
    const std::size_t d1, const std::size_t d2, const std::size_t d3);
// A : d2 * d1
// B : d2 * d3
// C : d1 * d3
__global__ void MatMulKernelATB(
    const float* A, const float* B, float* C,
    const std::size_t pitchA, const std::size_t pitchB, const std::size_t pitchC,
    const std::size_t d1, const std::size_t d2, const std::size_t d3);
// A : d1 * d2
// B : d3 * d2
// C : d1 * d3
__global__ void MatMulKernelABT(
    const float* A, const float* B, float* C,
    const std::size_t pitchA, const std::size_t pitchB, const std::size_t pitchC,
    const std::size_t d1, const std::size_t d2, const std::size_t d3);

#endif