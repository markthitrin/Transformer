#include "Config.h"
#include "Header.h"
#include "Linear.h"
#include "Tensor.h"
#include "Timer.h"
#include "Util.h"

Linear::Linear(const int inD, const int outD) : 
    weight(inD, outD), bias(1, outD), weightOpt(inD, outD), biasOpt(1, outD) {
        
    HeNormalInit(weight);
    HeNormalInit(bias);
}

void Linear::forward(TensorView input, TensorView output) {
    const int numT = std::min(numPar, 24);
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
    const int outD = bias.col;
    float* biasGrad = biasOpt.gradient.data;
    SetPar(inputGradient, 0.0f);
    const int numT = std::min(numPar, 24);
    #pragma omp parallel for num_threads(numT) reduction(+:biasGrad[:outD])
    for(int i = 0;i < batch * sequenceLength;i++) {
        for(int j = 0;j < outD;j++) {
            biasGrad[j] += outputGradient[i * outD + j];
        }
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