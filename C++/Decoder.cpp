#include "Config.h"
#include "Decoder.h"
#include "DecoderLayer.h"
#include "Embedding.h"
#include "Header.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "Softmax.h"
#include "Tensor.h"
#include "Util.h"

Decoder::Decoder() {
    out.reserve(N);
    gradient.reserve(N);
    for(int i = 0;i < N;i++) {
        out.emplace_back(batch * sequenceLength, dModel);
        gradient.emplace_back(batch * sequenceLength, dModel);
    }
}


void Decoder::forward(TensorView input, TensorView encoderOut, TensorView output, const int srcSeq[batch], const int tgtSeq[batch]) {
    layers[0].forward(input, encoderOut, out[0], srcSeq, tgtSeq);
    for(int i = 1;i < N;i++) {
        layers[i].forward(out[i - 1], encoderOut, out[i], srcSeq, tgtSeq);
    }
    norm.forward(out[N - 1], output);
}

void Decoder::predict(TensorView input, TensorView encoderOut, TensorView output, const int srcSeq[batch], const int tgtSeq[batch]) {
    layers[0].predict(input, encoderOut, out[0], srcSeq, tgtSeq);
    for(int i = 1;i < N;i++) {
        layers[i].predict(out[i - 1], encoderOut, out[i], srcSeq, tgtSeq);
    }
    norm.predict(out[N - 1], output);
}


void Decoder::backward(
    TensorView outputGradient, TensorView inputGradient, TensorView encoderGradient, TensorView encoderOut,
    const int srcSeq[batch], const int tgtSeq[batch]) {

    encoderGradient = 0.0; // need to be manually set.
    norm.backward(outputGradient, gradient[N - 1]);
    for(int i = N - 1;i >= 1;i--) {
        layers[i].backward(gradient[i], encoderGradient, gradient[i - 1], encoderOut, srcSeq, tgtSeq);
    }
    layers[0].backward(gradient[0], encoderGradient, inputGradient, encoderOut, srcSeq, tgtSeq);
}

void Decoder::updateParameter() {
    for(int i = 0;i < N;i++) {
        layers[i].updateParameter();
    }
    norm.updateParameter();
}