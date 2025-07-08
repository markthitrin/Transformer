#include "Header.h"
#include "Tensor.h"
#include "DecoderLayer.h"
#include "Softmax.h"
#include "Embedding.h"
#include "Linear.h"
#include "LayerNorm.h"
#include "Util.h"
#include "Decoder.h"

Decoder::Decoder() {
    out.reserve(N);
    gradient.reserve(N);
    for(int i = 0;i < N;i++) {
        out.emplace_back(batch * sequenceLength, dModel);
        gradient.emplace_back(batch * sequenceLength, dModel);
    }
}


void Decoder::forward(TensorView input, TensorView encoderOutput, TensorView output, const int srcSeq[batch], const int tgtSeq[batch]) {
    layers[0].forward(input, encoderOutput, out[0], srcSeq, tgtSeq);
    for(int i = 1;i < N;i++) {
        layers[i].forward(input, encoderOutput, out[0], srcSeq, tgtSeq);
    }
    norm.forward(out[N - 1], output);
}

void Decoder::predict(TensorView input, TensorView encoderOutput, TensorView output, const int srcSeq[batch], const int tgtSeq[batch]) {
    layers[0].predict(input, encoderOutput, out[0], srcSeq, tgtSeq);
    for(int i = 1;i < N;i++) {
        layers[i].predict(input, encoderOutput, out[0], srcSeq, tgtSeq);
    }
    norm.predict(out[N - 1], output);
}


void Decoder::backward(
    TensorView outputGradient, TensorView inputGradient, TensorView encoderGradient, TensorView encoderOutput,
    const int srcSeq[batch], const int tgtSeq[batch]) {

    norm.backward(outputGradient, gradient[N - 1]);
    for(int i = N - 1;i >= 1;i--) {
        layers[i].backward(gradient[i], encoderGradient, gradient[i - 1], encoderOutput, srcSeq, tgtSeq);
    }
    layers[0].backward(gradient[0], encoderGradient, inputGradient, encoderOutput, srcSeq, tgtSeq);
}

void Decoder::updateParameter() {
    for(int i = 0;i < N;i++) {
        layers[i].updateParameter();
    }
    norm.updateParameter();
}