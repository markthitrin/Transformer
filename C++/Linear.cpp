#include "Header.h"
#include "Tensor.h"
#include "Util.h"
#include "Linear.h"
#include "Timer.h"

Linear::Linear(const int in, const int out) : 
    weight(in, out), bias(1, out), weightOpt(in, out), biasOpt(1, out) {
        
    HeNormalInit(weight);
    HeNormalInit(bias);
}

void Linear::forward(TensorView input, TensorView output) {
    const int numT = getNumThreads(batch * sequenceLength, batch * sequenceLength * weight.col, 1, 2);
    #pragma omp parallel for num_threads(numT) schedule(static)
    for(int i = 0;i < batch * sequenceLength;i++) {
        output.sliceRow(i,1) = bias;
    }
    MatMulPlusABPar(input, weight, output);
    Timer::CheckPoint();
}

void Linear::predict(TensorView input, TensorView output) {
    return forward(input, output);
}

void Linear::backward(TensorView outputGradient, TensorView inputGradient, TensorView input) {
    inputGradient = 0;
    float* biasGrad = biasOpt.gradient.data;
    const int numT = getNumThreads(batch * sequenceLength, batch * sequenceLength * weight.col, 1, 2);
    #pragma omp parallel for num_threads(numT) reduction(+:biasGrad[:weight.col])
    for(int i = 0;i < batch * sequenceLength;i++) {
        Plus(biasOpt.gradient, outputGradient.sliceRow(i,1), biasOpt.gradient);
    }
    MatMulPlusATBPar(input, outputGradient, weightOpt.gradient);
    MatMulPlusABTPar(outputGradient, weight, inputGradient);
    Timer::CheckPoint();
}

void Linear::updateParameterTask() {
    #pragma omp task
    AdamOpt(weight, weightOpt);
    #pragma omp task
    AdamOpt(bias, biasOpt);
}

void Linear::checkUpdatedParam(cnpy::npz_t npFile, std::string prefix) {
    Tensor weightUpdated(weight.row, weight.col);
    Tensor biasUpdated(1, weight.col);

    weightUpdated.loadNp(npFile, prefix + ".updated_weight");
    biasUpdated.loadNp(npFile, prefix + ".updated_bias");

    PrintTestResult("backward " + prefix + ".weight", weight, weightUpdated);
    PrintTestResult("backward " + prefix + ".bias", bias, biasUpdated);
}