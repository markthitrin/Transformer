#ifndef UTIL
#define UTIL

#include "Header.h"
#include "Tensor.h"
#include "Config.h"

template<int row, int col>
class AdamOptGradient {
public:
    AdamOptGradient() : t(1) {
        gradient.init();
        accM.init();
        accV.init();
    }

    Tensor<row, col> gradient;
    Tensor<row, col> accM;
    Tensor<row, col> accV;
    int t;
};

template<int row, int col>
void AdamOpt(Tensor<row, col>& _param, AdamOptGradient<row, col>& _opt, int feedCount = 1) {
    IMPORT(param);
    IMPORTA(gradient, _opt.gradient);
    IMPORTA(accM, _opt.accM);
    IMPORTA(accV, _opt.accV);

    // const float learningRate = std::sqrt(dModel) * std::min(std::pow(_opt.t, -0.5), _opt.t * std::pow(warmupStep, -1.5)) / feedCount;
    const float learningRate = lr;
    const float invPowBeta1 = 1.0f / (1.0f - std::pow(beta1,_opt.t));
    const float invPowBeta2 = 1.0f / (1.0f - std::pow(beta2,_opt.t));
        for(int i = 0;i < row * col;i++) {
        accM[i] = accM[i] * beta1 + gradient[i] * (1.0f - beta1);
        accV[i] = accV[i] * beta2 + gradient[i] * gradient[i] * (1.0f - beta2);
        float mHat = accM[i] * invPowBeta1;
        float vHat = accV[i] * invPowBeta2;
        param[i] -= learningRate * mHat / (std::sqrt(vHat) + eps);
    }
    Reset(_opt.gradient);
    _opt.t++;
}

float ComputCrossEntropy(const float* logits, int target_token, float* grad) {
    float max_logit = logits[0];
    for (int i = 1; i < tgtVocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < tgtVocab; ++i) {
        sum_exp += std::exp(logits[i] - max_logit);
    }

    float loss = 0.0f;
    for (int i = 0; i < tgtVocab; ++i) {
        float prob = std::exp(logits[i] - max_logit) / sum_exp;
        grad[i] = prob;
        if (i == target_token) {
            loss = -std::log(prob + 1e-9f);
            grad[i] -= 1.0f;
        }
    }

    return loss;
}

float _CrossEntropy(Tensor<sequenceLength, tgtVocab> logits, float* targetToken, Tensor<sequenceLength, tgtVocab> grad, int npd) {
    float loss = 0.0f;
    for(int i = 0;i < npd + 1;i++) {
        loss += ComputCrossEntropy(logits.data + i * tgtVocab, targetToken[i], grad.data + i * tgtVocab);
    }
    return loss / (npd + 1);
}

float CrossEntropy(
    Tensor<batch * sequenceLength, tgtVocab> logits, 
    Tensor<1, batch * sequenceLength> target, 
    Tensor<batch * sequenceLength, tgtVocab> grad,
    int npd[batch]) {

    float loss = 0.0f;
    for(int i = 0;i < batch;i++) {
        loss += _CrossEntropy(
            logits.sliceRow<sequenceLength>(i * sequenceLength),
            target.data + i * sequenceLength,
            grad.sliceRow<sequenceLength>(i * sequenceLength),
            npd[i]
        );
    }
    return loss / batch;
}

void GetAnswer(Tensor<batch * sequenceLength, tgtVocab> logits, int output[batch * sequenceLength]) {
    for(int i = 0;i < batch * sequenceLength;i ++) {
        float* row = logits.data + i * tgtVocab;
        float maxValue = -FLT_MAX;
        int ind = 0;
        for(int j = 0;j < tgtVocab;j++) {
            if(maxValue < row[j]) {
                maxValue = row[j];
                ind = j;
            }
            if(j < 10) std::cout << row[j] << " ";
        }
        std::cout << std::endl;
        output[i] = ind;
    }
}

float fast_logf(float x) {
    union { float f; uint32_t i; } vx = { x };
    float y = vx.i;
    y *= 1.1920928955078125e-7f;
    return y - 127.0f;
}

#endif