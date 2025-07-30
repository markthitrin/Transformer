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
    #pragma omp parallel for num_threads(64) schedule(static)
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
    
    #pragma omp parallel for num_threads(64) reduction(+:biasGrad[:dModel], alphaGrad[:dModel])
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


void LayerNorm::loadParam(cnpy::npz_t npFile, std::string prefix) {
    alpha.loadNp(npFile, prefix + ".alpha");
    bias.loadNp(npFile, prefix + ".bias");
}

void LayerNorm::checkUpdatedParam(cnpy::npz_t npFile, std::string prefix) {
    Tensor alphaUpdated(1, dModel);
    Tensor biasUpdated(1, dModel);
    alphaUpdated.loadNp(npFile, prefix + ".updated_alpha");
    biasUpdated.loadNp(npFile, prefix + ".updated_bias");

    PrintTestResult("backward " + prefix + ".alpha", alpha, alphaUpdated);
    PrintTestResult("backward " + prefix + ".bias", bias, biasUpdated);
}


void LayerNorm::forwardTest(cnpy::npz_t npFile, std::string prefix) {
    Tensor target(batch * sequenceLength, dModel);
    Tensor input(batch * sequenceLength, dModel);
    Tensor output(batch * sequenceLength, dModel);

    input.loadNp(npFile, prefix + ".input");
    target.loadNp(npFile, prefix + ".output");

    forward(input, output);

    PrintTestResult("forward", output, target);
}

void LayerNorm::backwardTest(cnpy::npz_t npFile, std::string prefix) {
    Tensor outputGradient(batch * sequenceLength, dModel);
    Tensor inputGradient(batch * sequenceLength, dModel);
    Tensor input(batch * sequenceLength, dModel);
    Tensor output(batch * sequenceLength, dModel);
    outputGradient = 1.0f / outputGradient.row / outputGradient.col;

    input.loadNp(npFile, prefix + ".input");

    forward(input, output);
    backward(outputGradient, inputGradient);
    #pragma omp parallel
    {
        #pragma omp single
        {
            updateParameterTask();
            #pragma omp taskwait
        }
    }

    checkUpdatedParam(npFile, prefix);
}

