#include "Config.h"
#include "Header.h"
#include "Tensor.h"
#include "Softmax.h"
#include "Timer.h"

Softmax::Softmax() {

}

void Softmax::forward(TensorView input, TensorView output) {
    #pragma omp parallel for num_threads(numPar) schedule(static)
    for (int i = 0; i < batch * head * sequenceLength; i++) {
        float buffer[sequenceLength];
        float sumExp = 0.0;
        float maxValue = -FLT_MAX;
        for (int j = 0; j < sequenceLength; j++) {
            maxValue = std::max(maxValue, input[i * sequenceLength + j]);
        }

        for (int j = 0; j < sequenceLength; j++) {
            buffer[j] = input[i * sequenceLength + j] - maxValue;
            buffer[j] = expf(buffer[j]);
        }

        for(int j = 0;j < sequenceLength;j++) {
            sumExp += buffer[j];
        }

        for (int j = 0; j < sequenceLength; j++) {
            output[i * sequenceLength + j] = buffer[j] / sumExp;
        }
    }   
    // Timer::CheckPoint();
}

void Softmax::predict(TensorView input, TensorView output) {
    forward(input, output);
}

void Softmax::backward(TensorView outputGradient, TensorView inputGradient, TensorView output) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < batch * head * sequenceLength; i++) {
        float sumGY = 0.0f;

        for (int j = 0; j < sequenceLength; j++) { 
            sumGY += outputGradient[i * sequenceLength + j] * output[i * sequenceLength + j];
        }

        for (int j = 0; j < sequenceLength; j++) {
            inputGradient[i * sequenceLength + j] = output[i * sequenceLength + j] * (outputGradient[i * sequenceLength + j] - sumGY);
        }
    }
    // Timer::CheckPoint();
}
