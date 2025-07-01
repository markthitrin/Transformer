#ifndef TENSOR_FUNCTION
#define TENSOR_FUCNTION

#include "Header.cuh"
#include "Tensor.cuh"

void CopyAsync(Tensor& A, Tensor& B);
void CopyBatchAsync(Tensor& A, Tensor& B, const std::size_t batch);

void Plus(Tensor& A, Tensor& B, Tensor& C);
void Plus(Tensor& A, const float x, Tensor& C);
void PlusBatch(Tensor& A, Tensor& B, Tensor& C, const std::size_t batch);
void PlusInplaceBatchReduce(Tensor& A, Tensor& C, const std::size_t batch);

void Sub(Tensor& A, Tensor& B, Tensor& C);
void Sub(Tensor& A, const float x, Tensor& C); 

void Mul(Tensor& A, Tensor& B, Tensor& C);
void Mul(Tensor& A, const float x, Tensor& C);

void Div(Tensor& A, Tensor& B, Tensor& C);
void Div(Tensor& A, const float x, Tensor& C);

void Set(Tensor& A, const float x);
void Reset(Tensor& A);

void ReduceMax(Tensor& A, Tensor& C);
void ReduceSumExp(Tensor& input, Tensor& maxValue, Tensor& sumExp);

void GetPositionalEncode(Tensor& A);

void Print(Tensor& A);

void MatMulPlus(Tensor& A,Tensor& B, Tensor& C, bool ATransposed, bool BTransposed);
#endif