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
    for(int i = 0;i < batch * sequenceLength;i++) {
        output.sliceRow(i, 1) = table.sliceRow(input[i], 1);
    }
    Mul(output, std::sqrt(dModel), output);
    Timer::CheckPoint();
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
    Timer::CheckPoint();
}

void Embedding::updateParameter() {
    for(int i = 0;i < tableOpt.size();i++) {
        if(needUpdate[i]) {
            AdamOpt(table.sliceRow(i, 1), tableOpt[i]);
            needUpdate[i] = false;
        }
    }
}