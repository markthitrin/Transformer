#include "Config.h"
#include "Decoder.h"
#include "DecoderLayer.h"
#include "Header.h"
#include "LayerNorm.h"
#include "Tensor.h"
#include "Util.h"

Decoder::Decoder() {
    outi.reserve(N);
    gradient.reserve(N);
    for(int i = 0;i < N;i++) {
        outi.emplace_back(batch * sequenceLength, dModel);
        gradient.emplace_back(batch * sequenceLength, dModel);
    }
}


void Decoder::forward(
    TensorView input, TensorView encoderOut, TensorView output,
    const int srcSeq[batch], const int tgtSeq[batch]) {

    layers[0].forward(input, encoderOut, outi[0], srcSeq, tgtSeq);
    for(int i = 1;i < N;i++) {
        layers[i].forward(outi[i - 1], encoderOut, outi[i], srcSeq, tgtSeq);
    }
    norm.forward(outi[N - 1], output);
}

void Decoder::predict(
    TensorView input, TensorView encoderOut, TensorView output,
    const int srcSeq[batch], const int tgtSeq[batch]) {

    layers[0].predict(input, encoderOut, outi[0], srcSeq, tgtSeq);
    for(int i = 1;i < N;i++) {
        layers[i].predict(outi[i - 1], encoderOut, outi[i], srcSeq, tgtSeq);
    }
    norm.predict(outi[N - 1], output);
}


void Decoder::backward(
    TensorView outputGradient, TensorView inputGradient, TensorView encoderGradient, TensorView encoderOut,
    const int srcSeq[batch], const int tgtSeq[batch]) {

    SetPar(encoderGradient, 0.0f); // need to be set manually
    norm.backward(outputGradient, gradient[N - 1]);
    for(int i = N - 1;i >= 1;i--) {
        layers[i].backward(gradient[i], encoderGradient, gradient[i - 1], encoderOut, srcSeq, tgtSeq);
    }
    layers[0].backward(gradient[0], encoderGradient, inputGradient, encoderOut, srcSeq, tgtSeq);
}

void Decoder::updateParameterTask() {
    for(int i = 0;i < N;i++) {
        layers[i].updateParameterTask();
    }
    norm.updateParameterTask();
}