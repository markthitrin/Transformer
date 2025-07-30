#include "Config.h"
#include "Header.h"
#include "ReLU.h"
#include "ReLU.h"
#include "Timer.h"

ReLU::ReLU() {;}

void ReLU::forward(TensorView input, TensorView output) {
    #pragma omp parallel for num_threads(8) schedule(static)
    for(int i = 0;i < input.row * input.col;i++) {
        output[i] = input[i] >= 0 ? input[i] : 0;
    }
    Timer::CheckPoint();
}

void ReLU::predict(TensorView input, TensorView output) {
    return forward(input, output);
}

void ReLU::backward(TensorView outputGradient, TensorView inputGradient, TensorView input) {
    #pragma omp parallel for num_threads(8) schedule(static)
    for(int i = 0;i < outputGradient.row * outputGradient.col;i++) {
        outputGradient[i] = input[i] >= 0 ? outputGradient[i] : 0;
    }
    CopyPar(outputGradient, inputGradient);
    Timer::CheckPoint();
}