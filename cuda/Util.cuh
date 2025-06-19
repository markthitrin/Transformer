#ifndef UTIL
#define UTIL

#include "Header.h"
#include "Tensor.cuh"
#include "UtilKernel.cuh"

class AdamOptimizer {
public:
    AdamOptimizer(Tensor param) : 
        gradient(param.row, param.col),
        accM(param.row, param.col),
        accV(param.row, param.col),
        t(1) {;}
    AdamOptimizer(const std::size_t row, const std::size_t col) : 
        gradient(row, col),
        accM(row, col),
        accV(row, col),
        t(1) {;}
    AdamOptimizer(AdamOptimizer& other) :
        gradient(other.gradient),
        accM(other.accM),
        accV(other.accV),
        t(1) {;}

    Tensor gradient;
    Tensor accM;
    Tensor accV;
    int t;
};

__global__ void AdamOptKernel(
    const float* param, const float* gradient, const float* accM, const float* accV, const float t,
    const std::size_t pitchParam, const std::size_t pitchGrad, const std::size_t pitchAccM, const std::size_t pitchAccV,
    const std::size_t row, const std::size_t col) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    const float invPowBeta1 = __frcp_rn(1.0f - std::pow(beta1, t));
    const float invPowBeta2 = __frcp_rn(1.0f - std::pow(beta2, t));
    if(r < row && c < col) {
        *Get(accM, r, c, pitchAccM) = *Get(accM, r, c, pitchAccM) * beta1 + *Get(gradient, r, c, pitchGrad) * (1.0f - beta1);
        *Get(accV, r, c, pitchAccM) = *Get(accV, r, c, pitchAccM) * beta2 + *Get(gradient, r, c, pitchGrad) * *Get(gradient, r, c, pitchGrad) * (1.0f - beta2);
        float mHat = *Get(accM, r, c, pitchAccM) * invPowBeta1;
        float vHat = *Get(accV, r, c, pitchAccM) * invPowBeta2;
        *Get(param, r, c, pitchParam) -= lr * mHat / (std::sqrt(vHat) + eps);
    }
}
void AdamOpt(Tensor param, AdamOptimizer opt, const int feedCount = 1) {
    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(param.col, BLOCKSIZE), ceil(param.row, BLOCKSIZE));
    AdamOptKernel<<<gridDim, blockDim>>>(
        param.data, opt.gradient.data, opt.accM.data, opt.accV.data, opt.t,
        param.pitch, opt.gradient.pitch, opt.accM.pitch, opt.accV.pitch,
        param.row, param.col);
    Reset(param);
    opt.t++;
}

float CrossEntropy(Tensor logits, Tensor target, Tensor gradient, int npd[batch]) {
    // not implemented
}
float fast_logf(float x) {
    union { float f; uint32_t i; } vx = { x };
    float y = vx.i;
    y *= 1.1920928955078125e-7f;
    return y - 127.0f;
}

void PrintTestResult(std::string text, Tensor A, Tensor B) {
    float* _A = (float*)malloc(sizeof(float) * A.row * A.col);
    float* _B = (float*)malloc(sizeof(float) * B.row * B.col);
	A.toFloat(_A);
    B.toFloat(_B);

    float result = 0.0f;
    for(int i = 0;i < A.row;i++) {
        for(int j = 0;j < A.col;j++) {
            result += std::abs(_A[i * A.col + j] - _B[i * B.col + j]);
        }
    }

	std::cout << "Test result [" << text << "] : " << result / A.row / A.col << "\n";
    int count = 0;
	for(int i  = 0;count < 6 && i < A.row * A.col;i++) {
        if(std::abs(_A[i] - _B[i]) < 0.001) {continue;}
		std::cout << "\t\t" << _A[i] << " :: " << _B[i];
        std::cout << "\t(" << i / A.col << ", " << i % A.col << ")" << std::endl;
        count++;
	}
	std::cout << std::endl;

    std::free(_A);
    std::free(_B);
}


void PrintTestResultT(std::string text, Tensor A, Tensor B) {
    float* _A = (float*)malloc(sizeof(float) * A.row * A.col);
    float* _B = (float*)malloc(sizeof(float) * B.row * B.col);
	A.toFloat(_A);
    B.toFloat(_B);
    
	float result = 0.0f;
    for(int i = 0;i < A.row;i++) {
        for(int j = 0;j < A.col;j++) {
            result += std::abs(_A[i * A.col + j] - _B[j * A.row + i]);
        }
    }

	std::cout << "Test result [" << text << "] : " << result / A.row / A.col / batch << "\n";
    int count = 0;
    for(int i = 0;i < A.row;i++) {
        for(int j = 0;j < A.col;j++) {
            if(std::abs(_A[i * A.col + j] - _B[j * A.row + i]) < 0.001) {continue;}
            std::cout << "\t\t" << _A[i * A.col + j] << " :: " << _B[j * A.row + i];
            std::cout << "\t()" << i << ", " << j << ")" << std::endl;
            if(count == 6) break;
        }
        if(count == 6) break;
    }
    std::cout << std::endl;
}

cudaGraphNode_t SyncDependency(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes = {}) {
    if(dependencyNodes.size() > 1) {
        cudaGraphNode_t dependency = nullptr;
        cudaGraphAddEmptyNode(&dependency, graph, dependencyNodes.data(), dependencyNodes.size());
        return dependency;
    }
    else {
        return dependencyNodes[0];
    }
    return nullptr;
}


#endif