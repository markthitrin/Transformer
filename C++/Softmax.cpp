#include "Header.h"
#include "Tensor.h"
#include "Softmax.h"
#include "Timer.h"

Softmax::Softmax() {

}

void Softmax::forward(TensorView input, TensorView output) {
    const int row = output.row;
    const int col = output.col;
    for (int i = 0; i < row; i++) {

        float sumExp = 0.0;
        float maxValue = -FLT_MAX;
        for (int j = 0; j < col; j++) {
            maxValue = std::max(maxValue, input[i * col + j]);
        }

        for (int j = 0; j < col; j++) {
            sumExp += std::exp(input[i * col + j] - maxValue);
        }

        for (int j = 0; j < col; j++) {
            output[i * col + j] = std::exp(input[i * col + j] - maxValue) / sumExp;
        }
    }
    Timer::CheckPoint();
    if(verbose) std::cout << "Softmax" << std::endl;
}

void Softmax::predict(TensorView input, TensorView output) {
    forward(input, output);
}

void Softmax::backward(TensorView outputGradient, TensorView inputGradient, TensorView output) {
    const int row = inputGradient.row;
    const int col = inputGradient.col;
    for (int i = 0; i < col; i++) {
        float sumGY = 0.0f;

        for (int j = 0; j < col; j++) {
            sumGY += outputGradient[i * col + j] * output[i * col + j];
        }

        for (int j = 0; j < col; j++) {
            inputGradient[i * col + j] = output[i * col + j] * (outputGradient[i * col + j] - sumGY);
        }
    }
}
