#include "Header.h"
#include "Tensor.h"
#include "Softmax.h"
#include "Timer.h"
#include <immintrin.h>

Softmax::Softmax() {

}

void Softmax::forward(TensorView input, TensorView output) {
    const int row = output.row;
    const int col = output.col;
    
    const int numT = getNumThreads(batch * head * sequenceLength, batch * head * sequenceLength * sequenceLength * 0.017, 10, 1);
    if(verbose) std::cout << "Softmax : " << input.row << ", " << input.col << " " << numT << std::endl;
    #pragma omp parallel for num_threads(numT) schedule(static)
    for (int i = 0; i < row; i++) {
        float buffer[sequenceLength];
        float sumExp = 0.0;
        float maxValue = -FLT_MAX;
        for (int j = 0; j < col; j++) {
            maxValue = std::max(maxValue, input[i * col + j]);
        }

        for (int j = 0; j < col; j++) {
            buffer[j] = input[i * col + j] - maxValue;
            buffer[j] = expf(buffer[j]);
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
    #pragma omp parallel for schedule(static)
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
