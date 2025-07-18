#include "Header.h"
#include "Tensor.h"
#include "Util.h"
#include "LayerNorm.h"
#include "Timer.h"

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
    const int row =  output.row;
    const int col =  output.col;

    for (int i = 0; i < row; i++) {
        float mean = 0.0f;
        for (int j = 0; j < col; j++) {
            mean += input[i * col + j];
        }
        mean /= col;

        std[i] = 0;
        for (int j = 0; j < col; j++) {
            const float x = (input[i * col + j] - mean);
            std[i] += x * x;
        }
        std[i] /= (col - 1);
        std[i] = std::sqrt(std[i]);

        for (int j = 0; j < col; j++) {
            xHat[i * col + j] = (input[i * col + j] - mean) / (std[i] + eps);
            output[i * col + j] = alpha[j] * xHat[i * col + j] + bias[j];
        }
    }
    Timer::CheckPoint();
    if(verbose) std::cout << "LayerNorm" << std::endl;
}

void LayerNorm::predict(TensorView input, TensorView output) {
    forward(input, output);
}

void LayerNorm::backward(TensorView outputGradient, TensorView inputGradient) {
    constexpr int row = batch * sequenceLength;
    constexpr int col = dModel;

    const float invCol = 1.0f / col;
    for (int i = 0; i < row; i++) {
        const float invO = 1.0f / (std[i] + eps);
        float sumG = 0;
        float sumGXHat = 0;
        for (int j = 0; j < col; j++) {
            float gxH = outputGradient[i * col + j] * xHat[i * col + j];
            alphaOpt.gradient[j] += gxH;
            sumGXHat += gxH;
        }
        for (int j= 0;j< col;j++) {
            biasOpt.gradient[j] += outputGradient[i * col + j];
            sumG += outputGradient[i * col + j];
        }
        float a = invCol * sumG;
        float b = invCol * sumGXHat;
        for (int j = 0; j < col; j++) {
            inputGradient[i * col + j] = invO * (outputGradient[i * col + j] - a - xHat[i * col + j] * b) * alpha[j];
        }
    }
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
    updateParameter();

    checkUpdatedParam(npFile, prefix);
}

void LayerNorm::updateParameter() {
    AdamOpt(alpha, alphaOpt);
    AdamOpt(bias, biasOpt);
}
