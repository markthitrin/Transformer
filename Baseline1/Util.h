#ifndef UTIL
#define UTIL

#include "Header.h"

Matrix AdamOpt(Matrix& accM, Matrix& accV, const Matrix& gradient, const int t);
float CrossEntropy(const Tensor& output, const Tensor& target);
Tensor CrossEntropyGradient(const Tensor& output, const Tensor& target);
void XavierUniformInit(Matrix& W);
void UniformInit(Matrix& W, const float limit);
void HeNormalInit(Matrix& W);
int RandomInt(int low, int high);

#endif