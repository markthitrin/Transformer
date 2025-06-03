#ifndef UTIL
#define UTIL

#include "Header.h"
#include "Tensor.h"

template<int d,int col>
void AdamOpt(Tensor _param, Tensor _accM, Tensor _accV, const Tensor _gradient, const int t) {
    IMPORT_CONST(gradient);
    IMPORT(param);
    IMPORT(accM);
    IMPORT(accV);
    const float invPowBeta1 = 1.0f / (1.0f - std::pow(beta1, t));
    const float invPowBeta2 = 1.0f / (1.0f - std::pow(beta2, t));
    const float learningRate = std::sqrt(dModel) * std::min(std::pow(t, -0.5), t * std::pow(warmupStep, -1.5));
    for (int i = 0; i < d * col; i++) {
        accM[i] = accM[i] * beta1 + gradient[i] * (1.0f - beta1);
        accV[i] = accV[i] * beta2 + gradient[i] * gradient[i] * (1.0f - beta2);
        float mHat = accM[i] * invPowBeta1;
        float vHat = accV[i] * invPowBeta2;
        param[i] -= learningRate * mHat / (std::sqrt(vHat) + eps);

    }
}

template<int d,int col>
float CrossEntropy(const Tensor _output, const Tensor _target, Tensor _outGradient) {
    IMPORT_CONST(target);
    IMPORT(outGradient);
    IMPORT(output);
    Reset<d, col>(outGradient);
    float loss = 0;
    constexpr float invD = 1.0f / d;
    for (int i = 0; i < d; i++) {
        int targetToken = target[i];
        if (output[i * col + targetToken] < 1e-8) {
            outGradient[i * col + targetToken] = -1.0f / 1e-8 * invD;
        }
        else {
            outGradient[i * col + targetToken] = -1.0f / output[i * col + targetToken] * invD;
        }
    }
    for (int i = 0; i < d; i++) {
        int targetToken = target[i];
        loss += std::log(output[i * col + targetToken]);
    }
    loss *= -1.0f / d;
    return loss;
}

template<int d,int col>
void XavierUniformInit(Tensor _param) {
    IMPORT(param);
    constexpr int in = d;
    constexpr int out = col;
    float limit = std::sqrt(6.0f / (in + out));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (int i = 0; i < d * col; i++) {
        param[i] = dist(gen);
    }
}

template<int d,int col>
void UniformInit(Tensor _param, const float limit) {
    IMPORT(param);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (int i = 0; i < d * col; ++i) {
        param[i] = dist(gen);
    }
}

template<int d,int col,int in>
void HeNormalInit(Tensor _param) {
    std::random_device rd;
    std::mt19937 gen(rd());
    float stddev = std::sqrt(2.0f / in);
    std::normal_distribution<float> dist(0.0f, stddev);

    for (int i = 0; i < d * col; ++i) {
        _param[i] = dist(gen);
    }
}

float fast_logf(float x) {
    union { float f; uint32_t i; } vx = { x };
    float y = vx.i;
    y *= 1.1920928955078125e-7f;
    return y - 127.0f;
}

#endif