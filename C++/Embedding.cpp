#include "Config.h"
#include "Embedding.h"
#include "Header.h"
#include "Tensor.h"
#include "Timer.h"
#include "Util.h"

Embedding::Embedding(const int numTokens) : table(numTokens, dModel), needUpdate(numTokens, false) {
    tableOpt.reserve(numTokens);
    for(int i = 0;i < numTokens;i++) {
        tableOpt.emplace_back(1, dModel);
    }
    
    UniformInit(table, 0.1);
}

void Embedding::forward(const int input[batch * sequenceLength], TensorView output) {
    #pragma omp parallel for num_threads(4) schedule(static)
    for(int i = 0;i < batch * sequenceLength;i++) {
        output.sliceRow(i, 1) = table.sliceRow(input[i], 1);
    }
    MulPar(output, std::sqrt(dModel), output);
    Timer::CheckPoint();
}

void Embedding::predict(const int input[batch * sequenceLength], TensorView output) {
    return forward(input, output);
}

void Embedding::backward(TensorView outputGradient, const int* input) {
    MulPar(outputGradient, std::sqrt(dModel), outputGradient);
    for(int i = 0;i < batch * sequenceLength;i++) {
        needUpdate[input[i]] = true;
        tableOpt[input[i]].gradient += outputGradient.sliceRow(i, 1);
    }
    Timer::CheckPoint();
}

void Embedding::updateParameterTask() {
    #pragma omp task
    {
        for(int i = 0;i < tableOpt.size();i++) {
            if(needUpdate[i]) {
                AdamOpt(table.sliceRow(i, 1), tableOpt[i]);
                needUpdate[i] = false;
            }
        }
    }
}

void Embedding::loadParam(cnpy::npz_t npFile, std::string prefix) {
    table.loadNp(npFile, prefix + ".weight");
}

void Embedding::forwardTest(cnpy::npz_t npFile, std::string prefix) {
    int input[batch * sequenceLength];
    Tensor output(batch * sequenceLength, dModel);
    Tensor target(batch * sequenceLength, dModel);
    Tensor inputLoader(1, batch * sequenceLength);

    inputLoader.loadNp(npFile, prefix + ".input");
    for(int i = 0;i < batch * sequenceLength;i++) input[i] = inputLoader[i];
    target.loadNp(npFile, prefix + ".output");

    forward(input, output);

    PrintTestResult("forward", output, target);
}

void Embedding::checkUpdatedParam(cnpy::npz_t npFile, std::string prefix) {
    Tensor tableUpdated(srcVocab, dModel);

    tableUpdated.loadNp(npFile, prefix + ".updated_weights");

    PrintTestResult("backward table", table, tableUpdated);
}

void Embedding::backwardTest(cnpy::npz_t npFile, std::string prefix) {
    int input[batch * sequenceLength];
    Tensor inputLoader(1, batch * sequenceLength);
    Tensor outputGradient(batch * sequenceLength, dModel);
    Tensor output(batch * sequenceLength, dModel);

    outputGradient = 1.0f / outputGradient.row / outputGradient.col;
    inputLoader.loadNp(npFile, prefix + ".input");
    for(int i = 0;i < batch * sequenceLength;i++) input[i] = inputLoader[i];

    forward(input, output);
    backward(outputGradient, input);
    #pragma omp parallel
    {
        #pragma omp single 
        {
            updateParameterTask();
            #pragma omp taskwait
        }
    }
    checkUpdatedParam(npFile, prefix);
}