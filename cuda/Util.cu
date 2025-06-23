#include "Header.cuh"
#include "Tensor.cuh"
#include "TensorFunction.cuh"
#include "TensorKernel.cuh"
#include "TensorGraph.cuh"
#include "Util.cuh"
#include "UtilKernel.cuh"

AdamOptimizer::AdamOptimizer(Tensor param) : 
    gradient(param.row, param.col),
    accM(param.row, param.col),
    accV(param.row, param.col),
    t(1) {;}
AdamOptimizer::AdamOptimizer(const std::size_t row, const std::size_t col) : 
    gradient(row, col),
    accM(row, col),
    accV(row, col),
    t(1) {;}
AdamOptimizer::AdamOptimizer(const AdamOptimizer& other) :
    gradient(other.gradient),
    accM(other.accM),
    accV(other.accV),
    t(1) {;}
AdamOptimizer::~AdamOptimizer() {
    gradient.free();
    accM.free();
    accV.free();
}


void AdamOpt(Tensor param, AdamOptimizer opt, const std::size_t feedCount) {
    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(param.col, BLOCKSIZE), ceil(param.row, BLOCKSIZE));
    AdamOptKernel<<<gridDim, blockDim>>>(
        param.data, opt.gradient.data, opt.accM.data, opt.accV.data, opt.t,
        param.pitch, opt.gradient.pitch, opt.accM.pitch, opt.accV.pitch,
        feedCount, param.row, param.col);
}
cudaGraphNode_t AppendAdamOptNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor param, AdamOptimizer opt, std::size_t feedCount) {
    
    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(param.col, BLOCKSIZE), ceil(param.row, BLOCKSIZE));

    std::size_t* feedCountPtr = new std::size_t(feedCount);

    void* kernelArgs[] = { 
        &param.data, &opt.gradient.data, &opt.accM.data, &opt.accV.data, &opt.t,
        &param.pitch, &opt.gradient.pitch, &opt.accM.pitch, &opt.accV.pitch,
        &feedCountPtr, &param.row, &param.col
    };

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)AdamOptKernel;
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = nullptr;

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);

    return kernelNode;
}

float CrossEntropy(Tensor logits, Tensor target, Tensor gradient, int npd[batch]) {
    // not implemented
    return 0.0f;
}
float fast_logf(float x) {
    union { float f; uint32_t i; } vx = { x };
    float y = vx.i;
    y *= 1.1920928955078125e-7f;
    return y - 127.0f;
}

void Print(Tensor A, const std::size_t r0, const std::size_t c0, const std::size_t r, const std::size_t c) {
    float* _A = (float*)malloc(sizeof(float) * A.row * A.col);
	A.toFloat(_A);

    for(int i = 0;i < r;i++) {
        for(int j = 0;j < c;j++) {
            std::cout << _A[i * A.col + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
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

cudaGraphNode_t SyncDependency(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    if(dependencyNodes.size() > 1) {
        cudaGraphNode_t dependency = nullptr;
        cudaGraphAddEmptyNode(&dependency, graph, dependencyNodes.data(), dependencyNodes.size());
        return dependency;
    }
    else if(dependencyNodes.size()) {
        return dependencyNodes[0];
    }
    return nullptr;
}