#include "Header.h"
#include "Tensor.h"
#include "Util.h"
#include "Embedding.h"
#include "Timer.h"

Embedding::Embedding(const int numTokens) : table(numTokens, dModel), needUpdate(numTokens, false) {
    tableOpt.reserve(numTokens);
    for(int i = 0;i < numTokens;i++) {
        tableOpt.emplace_back(1, dModel);
    }
}

void Embedding::forward(const int input[batch * sequenceLength], TensorView output) {
    for(int i = 0;i < batch * sequenceLength;i++) {
        output.sliceRow(i, 1) = table.sliceRow(input[i], 1);
    }
    Mul(output, std::sqrt(dModel), output);
    Timer::CheckPoint();
    if(verbose) std::cout << "Embedding" << std::endl;
}

void Embedding::predict(const int input[batch * sequenceLength], TensorView output) {
    return forward(input, output);
}

void Embedding::backward(TensorView outputGradient, const int* input) {
    Mul(outputGradient, std::sqrt(dModel), outputGradient);
    for(int i = 0;i < batch * sequenceLength;i++) {
        needUpdate[input[i]] = true;
        tableOpt[input[i]].gradient += outputGradient.sliceRow(i, 1);
    }
}

void Embedding::updateParameter() {
    for(int i = 0;i < tableOpt.size();i++) {
        if(needUpdate[i]) {
            AdamOpt(table.sliceRow(i, 1), tableOpt[i]);
            needUpdate[i] = false;
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
    updateParameter();

    checkUpdatedParam(npFile, prefix);
}