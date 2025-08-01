#include "Config.h"
#include "Embedding.h"
#include "Encoder.h"
#include "EncoderLayer.h"
#include "Header.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "Softmax.h"
#include "Tensor.h"
#include "Util.h"

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

void Encoder::updateParameter() {
    for(int i = 0;i < N;i++) {
        layers[i].updateParameter();
    }
    norm.updateParameter();
}