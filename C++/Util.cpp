#include "Tensor.h"
#include "Util.h"
#include "Header.h"

AdamOptimizer::AdamOptimizer() {;}

AdamOptimizer::AdamOptimizer(TensorView tv) : 
    gradient(tv.row, tv.col), accM(tv.row, tv.col), accV(tv.row, tv.col), t(1) {;}

AdamOptimizer::AdamOptimizer(const int row, const int col) : 
    gradient(row, col), accM(row, col), accV(row, col), t(1) {;}


void AdamOpt(TensorView param, AdamOptimizer& opt) {
    const float learningRate = lr;
    const float invPowBeta1 = 1.0f / (1.0f - std::pow(beta1,opt.t));
    const float invPowBeta2 = 1.0f / (1.0f - std::pow(beta2,opt.t));
        for(int i = 0;i < param.row * param.col;i++) {
        opt.accM[i] = opt.accM[i] * beta1 + opt.gradient[i] * (1.0f - beta1);
        opt.accV[i] = opt.accV[i] * beta2 + opt.gradient[i] * opt.gradient[i] * (1.0f - beta2);
        float mHat = opt.accM[i] * invPowBeta1;
        float vHat = opt.accV[i] * invPowBeta2;
        param[i] -= learningRate * mHat / (std::sqrt(vHat) + eps);
    }
    opt.gradient = 0;
    opt.t++;
}

float ComputCrossEntropy(const float* logits, int target_token, float* grad) {
    return 0;
}

float fast_logf(float x) {
    union { float f; uint32_t i; } vx = { x };
    float y = vx.i;
    y *= 1.1920928955078125e-7f;
    return y - 127.0f;
}
