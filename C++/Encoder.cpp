#include "Header.h"
#include "Tensor.h"
#include "EncoderLayer.h"
#include "Softmax.h"
#include "Embedding.h"
#include "Linear.h"
#include "LayerNorm.h"
#include "Util.h"
#include "Encoder.h"

Encoder::Encoder() {
    out.reserve(N);
    gradient.reserve(N);
    for(int i = 0;i < N;i++) {
        out.emplace_back(batch * sequenceLength, dModel);
        gradient.emplace_back(batch * sequenceLength, dModel);
    }
}


void Encoder::forward(TensorView input, TensorView output, const int srcSeq[batch]) {
    layers[0].forward(input, out[0], srcSeq);
    for(int i = 1;i < N;i++) {
        layers[i].forward(out[i - 1], out[i], srcSeq);
    }
    norm.forward(out[N - 1], output);
}

void Encoder::predict(TensorView input, TensorView output, const int srcSeq[batch]) {
    layers[0].predict(input, out[0], srcSeq);
    for(int i = 1;i < N;i++) {
        layers[i].predict(out[i - 1], out[i], srcSeq);
    }
    norm.predict(out[N - 1], output);
}


void Encoder::backward(TensorView outputGradient, TensorView inputGradient, const int srcSeq[batch]) {
    norm.backward(outputGradient, gradient[N - 1]);
    for(int i = N - 1;i >= 1;i--) {
        layers[i].backward(gradient[i], gradient[i - 1], srcSeq);
    }
    layers[0].backward(gradient[0], inputGradient, srcSeq);
}

void Encoder::updateParameterTask() {
    for(int i = 0;i < N;i++) {
        layers[i].updateParameterTask();
    }
    norm.updateParameterTask();
}

void Encoder::loadParam(cnpy::npz_t npFile, std::string prefix) {
    for(int i = 0;i < N;i++) {
        layers[i].loadParam(npFile, prefix + ".layer" + std::to_string(i));
    }
    norm.loadParam(npFile, prefix + ".norm");
}

void Encoder::checkUpdatedParam(cnpy::npz_t npFile, std::string prefix) {
    for(int i = 0;i < N;i++) {
        layers[i].checkUpdatedParam(npFile, prefix + ".layer" + std::to_string(i));
    }
    norm.checkUpdatedParam(npFile, prefix + ".norm");
}

void Encoder::forwardTest(cnpy::npz_t npFile, std::string prefix) {
    Tensor target(batch * sequenceLength, dModel);
    Tensor input(batch * sequenceLength, dModel);
    Tensor output(batch * sequenceLength, dModel);
    Tensor npdLoad(1,1);
    int seq[batch];

    input.loadNp(npFile, prefix + ".input");
    target.loadNp(npFile, prefix + ".output");
    npdLoad.loadNp(npFile, prefix + ".npd");
    for(int  i = 0; i < batch;i++) {
        seq[i] = npdLoad[0];
    }

    forward(input, output, seq);

    PrintTestResult("forward", output, target);
}

void Encoder::backwardTest(cnpy::npz_t npFile, std::string prefix) {
    Tensor target(batch * sequenceLength, dModel);
    Tensor input(batch * sequenceLength, dModel);
    Tensor output(batch * sequenceLength, dModel);
    Tensor outputGradient(batch * sequenceLength, dModel);
    Tensor inputGradient(batch * sequenceLength, dModel);
    Tensor npdLoad(1,1);
    int seq[batch];

    outputGradient = 1.0f / outputGradient.row / outputGradient.col;

    input.loadNp(npFile, prefix + ".input");
    npdLoad.loadNp(npFile, prefix + ".npd");
    for(int  i = 0; i < batch;i++) {
        seq[i] = npdLoad[0];
    }

    forward(input, output, seq);
    backward(outputGradient, inputGradient, seq);
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