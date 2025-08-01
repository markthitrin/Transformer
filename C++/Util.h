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

float CrossEntropy(TensorView logits, const int* target_token, const int* tgtSeq, TensorView grad);

float fast_logf(float x);

#endif