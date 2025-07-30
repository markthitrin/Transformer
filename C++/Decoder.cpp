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

void Decoder::loadParam(cnpy::npz_t npFile, std::string prefix) {
    for(int i = 0;i < N;i++) {
        layers[i].loadParam(npFile, prefix + ".layer" + std::to_string(i));
    }
    norm.loadParam(npFile, prefix + ".norm");
}

void Decoder::checkUpdatedParam(cnpy::npz_t npFile, std::string prefix) {
    for(int i = 0;i < N;i++) {
        layers[i].checkUpdatedParam(npFile, prefix + ".layer" + std::to_string(i));
    }
    norm.checkUpdatedParam(npFile, prefix + ".norm");
}

void Decoder::forwardTest(cnpy::npz_t npFile, std::string prefix) {
    Tensor target(batch * sequenceLength, dModel);
    Tensor input1(batch * sequenceLength, dModel);
    Tensor input2(batch * sequenceLength, dModel);
    Tensor output(batch * sequenceLength, dModel);
    Tensor npdLoad(1,2);
    int srcSeq[batch];
    int tgtSeq[batch];

    input1.loadNp(npFile, prefix + ".input1");
    input2.loadNp(npFile, prefix + ".input2");
    target.loadNp(npFile, prefix + ".output");
    npdLoad.loadNp(npFile, prefix + ".npd");
    for(int i = 0;i < batch;i++) srcSeq[i] = npdLoad[0];
    for(int i = 0;i < batch;i++) tgtSeq[i] = npdLoad[1];

    forward(input1, input2, output, srcSeq, tgtSeq);

    PrintTestResult("forward", output, target);
}

void Decoder::backwardTest(cnpy::npz_t npFile, std::string prefix) {
    Tensor target(batch * sequenceLength, dModel);
    Tensor input1(batch * sequenceLength, dModel);
    Tensor input2(batch * sequenceLength, dModel);
    Tensor output(batch * sequenceLength, dModel);
    Tensor outputGradient(batch * sequenceLength, dModel);
    Tensor inputGradient(batch * sequenceLength, dModel);
    Tensor npdLoad(1,2);
    int srcSeq[batch];
    int tgtSeq[batch];

    outputGradient = 1.0f / outputGradient.row / outputGradient.col;
    input1.loadNp(npFile, prefix + ".input1");
    input2.loadNp(npFile, prefix + ".input2");
    target.loadNp(npFile, prefix + ".output");
    npdLoad.loadNp(npFile, prefix + ".npd");
    for(int i = 0;i < batch;i++) srcSeq[i] = npdLoad[0];
    for(int i = 0;i < batch;i++) tgtSeq[i] = npdLoad[1];

    forward(input1, input2, output, srcSeq, tgtSeq);
    backward(outputGradient, inputGradient, inputGradient, input2, srcSeq, tgtSeq);
    #pragma omp parallel
    {
        #pragma omp single
        {
            updateParameterTask();
            #pragma taskwait
        }
    }

    checkUpdatedParam(npFile, prefix);
}