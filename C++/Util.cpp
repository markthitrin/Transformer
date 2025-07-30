#include "Tensor.h"
#include "Util.h"
#include "Header.h"
#include "Timer.h"

AdamOptimizer::AdamOptimizer() {;}

AdamOptimizer::AdamOptimizer(TensorView tv) : 
    gradient(tv.row, tv.col), accM(tv.row, tv.col), accV(tv.row, tv.col), t(1) {;}

AdamOptimizer::AdamOptimizer(const int row, const int col) : 
    gradient(row, col), accM(row, col), accV(row, col), t(1) {;}

void AdamOpt(TensorView param, AdamOptimizer& opt) {
    const float learningRate = lr;
    const float PowBeta1 = (1.0f - std::pow(beta1,opt.t));
    const float PowBeta2 = (1.0f - std::pow(beta2,opt.t));
    for(int i = 0;i < param.row * param.col;i++) {
        opt.accM[i] = opt.accM[i] * beta1 + opt.gradient[i] * (1.0f - beta1);
        opt.accV[i] = opt.accV[i] * beta2 + opt.gradient[i] * opt.gradient[i] * (1.0f - beta2);
        float mHat = opt.accM[i] / (PowBeta1 + eps);
        float vHat = opt.accV[i] / (PowBeta2 + eps);
        param[i] -= learningRate * mHat / (std::sqrt(vHat) + eps);
    }
    opt.gradient = 0;
    opt.t++;
}

float ComputCrossEntropy(TensorView logits, int target_token, TensorView grad) {
    float max_logit = logits[0];
    float buffer[tgtVocab];
    for (int i = 1; i < tgtVocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < tgtVocab; ++i) {
        buffer[i] = std::exp(logits[i] - max_logit);
        sum_exp += buffer[i];
    }

    float loss = 0.0f;
    for (int i = 0; i < tgtVocab; ++i) {
        grad[i] = buffer[i] / sum_exp;
    }
    loss = -std::log2(buffer[target_token] + eps);
    grad[target_token] -= 1.0f;

    return loss;
}

float _CrossEntropy(TensorView logits, const int* targetToken, int tgtSeq, TensorView grad) {
    float loss = 0.0f;
    const int numT = std::min(numPar, 9);
    #pragma omp parallel for num_threads(numT) schedule(static) reduction(+:loss)
    for(int i = 0;i < tgtSeq;i++) {
        loss += ComputCrossEntropy(logits.sliceRow(i, 1), targetToken[i], grad.sliceRow(i, 1));
    }
    return loss / tgtSeq;
}

float CrossEntropy(TensorView logits, const int* target_token, const int* tgtSeq, TensorView grad) {
    float loss = 0.0f;
    #pragma omp parallel for schedule(static) reduction(+:loss)
    for(int i = 0;i < batch;i++) {
        loss += _CrossEntropy(
            logits.sliceRow(i * sequenceLength, sequenceLength),
            target_token + i * sequenceLength,
            tgtSeq[i],
            grad.sliceRow(i * sequenceLength, sequenceLength));
    }
    return loss / batch;
}


float fast_logf(float x) {
    union { float f; uint32_t i; } vx = { x };
    float y = vx.i;
    y *= 1.1920928955078125e-7f;
    return y - 127.0f;
}

void PrintTestResult(std::string text, TensorView A, TensorView B) {
	float result = 0.0f;
	for(int i = 0;i < A.row * A.col;i++) {
		result += std::abs(A.data[i] - B.data[i]);
	}

	std::cout << "Test result [" << text << "] : " << result / A.row / A.col << "\n";
    int count = 0;
	for(int i  = 0;count < 6 && i < A.row * A.col;i++) {
        if(std::abs(A.data[i] - B.data[i]) < 0.00009) {continue;}
		std::cout << "\t\t" << A.data[i] << " :: " << B.data[i];
        std::cout << "\t(" << i / A.col << ", " << i % A.col << ")" << std::endl;
        count++;
	}
	std::cout << std::endl;
}

void PrintTestResultT(std::string text, TensorView A, TensorView B) {
	float result = 0.0f;
    for(int i = 0;i < batch;i++) {
        for(int j = 0;j < A.row;j++) {
            for(int k = 0;k < A.col;k++) {
                result += std::abs(A.data[i * (A.row * A.col) + j * (A.col) + k] - B.data[i*(A.col * A.row) + k * (A.row) + j]);
            }
        }
    }

	std::cout << "Test result [" << text << "] : " << result / A.row / A.col / batch << "\n";
	int count = 6;
    for(int i = 0;i < batch;i++) {
        for(int j = 0;j < A.row;j++) {
            for(int k = 0;k < A.col;k++) {
                if(std::abs(A.data[i * (A.row * A.col) + j * (A.col) + k] - B.data[i*(A.col * A.row) + k * (A.row) + j]) < 0.00009) continue;
                std::cout << "\t\t" << A.data[i * (A.row * A.col) + j * (A.col) + k]  << " :: " <<  B.data[i*(A.col * A.row) + k * (A.row) + j];
                std::cout << "\t(" <<j << ", " << k << ")" << std::endl;
                count--;
                if(count == 0) break;
            }
            if(count == 0) break;
        }
        if(count == 0) break;
    }

	std::cout << std::endl;
}
