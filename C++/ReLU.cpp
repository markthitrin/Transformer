#include "Header.h"
#include "ReLU.h"
#include "Config.h"
#include "ReLU.h"
#include "Timer.h"

ReLU::ReLU() : mask(batch * sequenceLength, dFF) {;}

void ReLU::forward(TensorView input, TensorView output) {
    for(int i = 0;i < input.row * input.col;i++) {
        mask[i] = input[i] >= 0;
        output[i] = input[i] * mask[i];
    }
    Timer::CheckPoint();
    if(verbose) std::cout << "ReLU" << std::endl;
}

void ReLU::predict(TensorView input, TensorView output) {
    return forward(input, output);
}

void ReLU::backward(TensorView outputGradient, TensorView inputGradient) {
    for(int i = 0;i < outputGradient.row * outputGradient.col;i++) {
        inputGradient[i] = mask[i] * outputGradient[i];
    }
}