#include "Header.h"
#include "Tensor.h"
#include "Util.h"
#include "LayerNorm.h"

LayerNorm::LayerNorm() : 
    alpha(1, dModel),
    bias(1, dModel),
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
}

void LayerNorm::predict(TensorView input, TensorView output) {
    forward(input, output);
}

void LayerNorm::backward(TensorView outputGradient, TensorView inputGradient) {
    const int row =  inputGradient.row;
    const int col =  inputGradient.col;

    const float invCol = 1.0f / col;
    for (int i = 0; i < row; i++) {
        const float invO = 1.0f / (std[i] + eps);
        float sumG = 0;
        float sumGXHat = 0;
        for (int j = 0; j < col; j++) {
            float gxH = outputGradient[i * col + j] * xHat[i * col + j];
            alphaOpt.gradient[j] += gxH;
            biasOpt.gradient[j] += outputGradient[i * col + j];
            sumG += outputGradient[i * col + j];
            sumGXHat += gxH;
        }
        float a = invCol * sumG;
        float b = invCol * sumGXHat;
        for (int j = 0; j < col; j++) {
            inputGradient[i * col + j] = invO * (outputGradient[i * col + j] - a - xHat[i * col + j] * b) * alpha[j];
        }
    }
}

void LayerNorm::updateParameter() {
    AdamOpt(alpha, alphaOpt);
    AdamOpt(bias, biasOpt);
}
