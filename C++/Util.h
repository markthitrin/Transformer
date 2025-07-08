#ifndef UTIL
#define UTIL

#include "Header.h"
#include "Tensor.h"
#include "Config.h"

class AdamOptimizer {
public:
    AdamOptimizer();
    AdamOptimizer(TensorView tv);
    AdamOptimizer(const int row, const int col);
    Tensor gradient;
    Tensor accM;
    Tensor accV;
    int t;
};


void AdamOpt(TensorView param, AdamOptimizer& opt);

float ComputCrossEntropy(const float* logits, int target_token, float* grad);

float fast_logf(float x);

void PrintTestResult(std::string text, TensorView A, TensorView B);

void PrintTestResultT(std::string text, TensorView A, TensorView B);

#endif