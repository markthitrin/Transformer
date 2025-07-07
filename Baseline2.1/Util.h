#ifndef UTIL
#define UTIL

#include "Header.h"
#include "Tensor.h"

template<int row, int col>
class AdamOptGradient {
public:
    AdamOptGradient() : t(1) {;}

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

template<int row,int col>
float CrossEntropy(const Tensor<row, col>& _output, const Tensor<row, col>& _target, Tensor<row, col>& _outGradient) {
    IMPORT_CONST(output);
    IMPORT_CONST(target);
    IMPORT(outGradient);

    Reset(_outGradient);
    float loss = 0;
    const float invRow = 1.0f / row;
    for (int i = 0; i < row; i++) {
        int targetToken = target[i];
        if (output[i*(col) + targetToken] < 1e-8) {
            outGradient[i*(col) + targetToken] = -1.0f / 1e-8 * invRow;
        }
        else {
            outGradient[i*(col) + targetToken] = -1.0f / output[i*(col) + targetToken] * invRow;
        }
        loss += std::log(output[i*(col) + targetToken]);
    }
    loss *= -invRow;
    return loss;
}

float fast_logf(float x) {
    union { float f; uint32_t i; } vx = { x };
    float y = vx.i;
    y *= 1.1920928955078125e-7f;
    return y - 127.0f;
}

template<int row,int col>
void PrintTestResult(std::string text, Tensor<row, col>& A, Tensor<row, col>& B) {
	float result = 0.0f;
	for(int i = 0;i < row * col;i++) {
		result += std::abs(A.data[i] - B.data[i]);
	}

	std::cout << "Test result [" << text << "] : " << result / row / col << "\n";
    int count = 0;
	for(int i  = 0;count < 6 && i < row * col;i++) {
        if(std::abs(A.data[i] - B.data[i]) < 0.001) {continue;}
		std::cout << "\t\t" << A.data[i] << " :: " << B.data[i];
        std::cout << "\t(" << i / col << ", " << i % col << ")" << std::endl;
        count++;
	}
	std::cout << std::endl;
}

template<int batch, int row,int col>
void PrintTestResultT(std::string text, Tensor<batch * row, col>& A, Tensor<batch * col, row>& B) {
	float result = 0.0f;
    for(int i = 0;i < batch;i++) {
        for(int j = 0;j < row;j++) {
            for(int k = 0;k < col;k++) {
                result += std::abs(A.data[i * (row * col) + j * (col) + k] - B.data[i*(col * row) + k * (row) + j]);
            }
        }
    }

	std::cout << "Test result [" << text << "] : " << result / row / col / batch << "\n";
	int count = 6;
    for(int i = 0;i < batch;i++) {
        for(int j = 0;j < row;j++) {
            for(int k = 0;k < col;k++) {
                if(std::abs(A.data[i * (row * col) + j * (col) + k] - B.data[i*(col * row) + k * (row) + j]) < 0.001) continue;
                std::cout << "\t\t" << A.data[i * (row * col) + j * (col) + k]  << " :: " <<  B.data[i*(col * row) + k * (row) + j];
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


#endif