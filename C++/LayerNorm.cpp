#include "Header.h"
#include "LayerNorm.h"
#include "Tensor.h"
#include "Timer.h"
#include "Util.h"

LayerNorm::LayerNorm() : 
    alpha(1, dModel),
    bias(1, dModel),

    alphaOpt(1, dModel),
    biasOpt(1, dModel),

    xHat(batch * sequenceLength, dModel),
    std(1, batch * sequenceLength) {
        
    alpha = 1;
    bias = 0;
}

void LayerNorm::forward(TensorView input, TensorView output) {
    const int numT = std::min(numPar, 52);
    #pragma omp parallel for num_threads(numT) schedule(static)
    for (int i = 0; i < batch  * sequenceLength; i++) {
        float mean = 0.0f;
        for (int j = 0; j < dModel; j++) {
            mean += input[i * dModel + j];
        }
        mean /= dModel;

        std[i] = 0;
        for (int j = 0; j < dModel; j++) {
            const float x = (input[i * dModel + j] - mean);
            std[i] += x * x;
        }
        std[i] /= (dModel - 1);
        std[i] = std::sqrt(std[i]);

        for (int j = 0; j < dModel; j++) {
            xHat[i * dModel + j] = (input[i * dModel + j] - mean) / (std[i] + eps);
            output[i * dModel + j] = alpha[j] * xHat[i * dModel + j] + bias[j];
        }
    }
    Timer::CheckPoint();
}

void LayerNorm::predict(TensorView input, TensorView output) {
    forward(input, output);
}

void LayerNorm::backward(TensorView outputGradient, TensorView inputGradient) {
    const float invDModel = 1.0f / dModel;
    float* biasGrad = biasOpt.gradient.data;
    float* alphaGrad = alphaOpt.gradient.data;

    const int numT = std::min(numPar, 52);
    #pragma omp parallel for num_threads(numT) reduction(+:biasGrad[:dModel], alphaGrad[:dModel])
    for (int i = 0; i < batch * sequenceLength; i++) {
        const float invO = 1.0f / (std[i] + eps);
        float sumG = 0;
        float sumGXHat = 0;
        for (int j = 0; j < dModel; j++) {
            float gxH = outputGradient[i * dModel + j] * xHat[i * dModel + j];
            alphaGrad[j] += gxH;
            sumGXHat += gxH;
        }
        for (int j= 0;j< dModel;j++) {
            biasGrad[j] += outputGradient[i * dModel + j];
            sumG += outputGradient[i * dModel + j];
        }
        float a = invDModel * sumG;
        float b = invDModel * sumGXHat;
        for (int j = 0; j < dModel; j++) {
            inputGradient[i * dModel + j] = invO * (outputGradient[i * dModel + j] - a - xHat[i * dModel + j] * b) * alpha[j];
        }
    }
    Timer::CheckPoint();
}

void LayerNorm::updateParameterTask() {
    #pragma omp task
    AdamOpt(alpha, alphaOpt);
    #pragma omp task
    AdamOpt(bias, biasOpt);
}