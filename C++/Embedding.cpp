#include "Header.h"
#include "Tensor.h"
#include "Util.h"
#include "Embedding.h"


Embedding::Embedding(const int numTokens) : table(numTokens, dModel), needUpdate(numTokens, false) {
    tableOpt.reserve(numTokens);
    for(int i = 0;i < numTokens;i++) {
        tableOpt.emplace_back(1, dModel);
    }
}

void Embedding::forward(const int input[batch * sequenceLength], TensorView output) {
    for(int i = 0;i < batch * sequenceLength;i++) {
        output.sliceRow(i, 1) = table[input[i]];
    }
    Mul(output, std::sqrt(dModel), output);
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
        }
    }
}
