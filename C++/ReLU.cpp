#include "Header.h"
#include "ReLU.h"
#include "Config.h"
#include "ReLU.h"
#include "Timer.h"

ReLU::ReLU() {;}

void ReLU::forward(TensorView input, TensorView output) {
    for(int i = 0;i < input.row * input.col;i++) {
        output[i] = input[i] >= 0 ? input[i] : 0;
    }
    Timer::CheckPoint();
    if(verbose) std::cout << "ReLU" << std::endl;
}

void ReLU::predict(TensorView input, TensorView output) {
    return forward(input, output);
}

void ReLU::backward(TensorView outputGradient, TensorView inputGradient, TensorView input) {
    for(int i = 0;i < outputGradient.row * outputGradient.col;i++) {
        inputGradient[i] = input[i] >= 0 ? outputGradient[i] : 0;
    }
}