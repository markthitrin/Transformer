#include "Header.h"
#include "ReLU.h"
#include "Config.h"
#include "ReLU.h"
#include "Timer.h"

ReLU::ReLU() {;}

void ReLU::forward(TensorView input, TensorView output) {
    const int numT = getNumThreads(batch * sequenceLength, batch * sequenceLength * dFF * 0.00085, 1, 1);
    if(verbose) std::cout << "ReLU : " << output.row << ", " << output.col << " " << numT << std::endl;
    #pragma omp parallel for num_threads(numT) schedule(static)
    for(int i = 0;i < input.row * input.col;i++) {
        output[i] = input[i] >= 0 ? input[i] : 0;
    }
    Timer::CheckPoint();
}

void ReLU::predict(TensorView input, TensorView output) {
    return forward(input, output);
}

void ReLU::backward(TensorView outputGradient, TensorView inputGradient, TensorView input) {
    const int numT = getNumThreads(batch * sequenceLength, batch * sequenceLength * dFF * 0.00085, 1, 1);
    if(verbose) std::cout << "ReLU : " << outputGradient.row << ", " << outputGradient.col << " " << numT << std::endl;
    #pragma omp parallel for num_threads(numT) schedule(static)
    for(int i = 0;i < outputGradient.row * outputGradient.col;i++) {
        outputGradient[i] = input[i] >= 0 ? outputGradient[i] : 0;
    }
    CopyPar(outputGradient, inputGradient);
    Timer::CheckPoint();
}