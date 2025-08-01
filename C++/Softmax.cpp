#include "Config.h"
#include "Header.h"
#include "Softmax.h"
#include "Tensor.h"
#include "Timer.h"

Softmax::Softmax() {;}

void Softmax::forward(TensorView input, TensorView output) {
    const int row = output.row;
    const int col = output.col;
    float buffer[sequenceLength];
    for (int i = 0; i < row; i++) {

        float sumExp = 0.0;
        float maxValue = -FLT_MAX;
        for (int j = 0; j < col; j++) {
            maxValue = std::max(maxValue, input[i * col + j]);
        }

        for (int j = 0; j < col; j++) {
            buffer[j] = input[i * col + j] - maxValue;
            buffer[j] = std::expf(buffer[j]);
        }

        for(int j = 0;j < col;j++) {
            sumExp += buffer[j];
        }

        for (int j = 0; j < col; j++) {
            output[i * col + j] = buffer[j] / sumExp;
        }
    }   
    Timer::CheckPoint();
}

void Softmax::predict(TensorView input, TensorView output) {
    forward(input, output);
}

void Softmax::backward(TensorView outputGradient, TensorView inputGradient, TensorView output) {
    const int row = inputGradient.row;
    const int col = inputGradient.col;
    for (int i = 0; i < row; i++) {
        float sumGY = 0.0f;

        for (int j = 0; j < col; j++) { 
            sumGY += outputGradient[i * col + j] * output[i * col + j];
        }

        for (int j = 0; j < col; j++) {
            inputGradient[i * col + j] = output[i * col + j] * (outputGradient[i * col + j] - sumGY);
        }
    }
    Timer::CheckPoint();
}
