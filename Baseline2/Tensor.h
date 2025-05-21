#ifndef TENSOR
#define TENSOR

#include "Header.h"

using Tensor = const std::unique_ptr<__m256[]>;
using Element = __m256;

int getSize(const int d, const int row, const int col);
Tensor create(const int d, const int row, const int col);
float* toFloat(const int size, Tensor a);

void plus(const int size, Tensor a, Tensor b, Tensor& c);
void plus(const int size, Tensor a, const float x, Tensor& c);
void sub(const int size, Tensor a, Tensor b, Tensor& c);
void sub(const int size, Tensor a, const float x, Tensor& c);
void mul(const int size, Tensor a, Tensor b, Tensor& c);
void mul(const int size, Tensor a, const float x, Tensor& c);
void div(const int size, Tensor a, Tensor b, Tensor& c);
void div(const int size, Tensor a, const float x, Tensor& c);

void MatMulAb(const int d1, const int d2, const int d3, Tensor a, Tensor b, Tensor& c);
void MatMulATb(const int d1, const int d2, const int d3, Tensor a, Tensor b, Tensor& c);
void MatMulAbT(const int d1, const int d2, const int d3, Tensor a, Tensor b, Tensor& c);

#endif