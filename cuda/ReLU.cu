#include "Header.h"
#include "Tensor.cuh"
#include "TensorGraph.cuh"
#include "Util.cuh"
#include "UtilKernel.cuh"
#include "ReLU.cuh"

ReLU::ReLU(
	Tensor input,
    Tensor output,
    Tensor outputGradient,
    Tensor inputGradient) noexcept :
    input(input),
    output(output),
    outputGradient(outputGradient),
    inputGradient(inputGradient)  { ; }

cudaGraphNode_t ReLU::AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes = {}) {
    cudaGraphNode_t k1 = AppendReLUNode(graph, dependencyNodes, input, output);
    return k1;
}

cudaGraphNode_t ReLU::AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes = {}) {
    return AppendGraphForward(graph, dependencyNodes);
}

cudaGraphNode_t ReLU::AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes = {}) {
    cudaGraphNode_t k1 = AppendReLUBackwardNode(graph, dependencyNodes, input, output);
}

__global__ void ReLUKernel(
	const float* A, const float* C,
	const std::size_t pitchA, const std::size_t pitchC,
	const std::size_t row, const std::size_t col) {

	const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

	if(r < row && c < col) {
		*Get(C, r, c, pitchC) = fmaxf(*Get(A, r, c, pitchA), 0.0f); 
	}
}

__global__ void ReLUBackwardKernel(
	const float* A, const float* B, const float* C,
	const std::size_t pitchA, const std::size_t pitchB, const std::size_t pitchC,
	const std::size_t row, const std::size_t col) {
	
	const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

	if(r < row && c < col) {
		*Get(C, r, c, pitchC) = float(*Get(B, r, c, pitchB) > 0) * *Get(A, r, c, pitchA); 
	}
}

cudaGraphNode_t AppendReLUNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes = {},
    Tensor A, Tensor C) {
    
    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);

    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(A.col, BLOCKSIZE), ceil(A.row, BLOCKSIZE));

    void* kernelArgs[] = { &A.data, &C.data, &A.pitch, &C.pitch, &A.row, &A.col};

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)ReLUKernel;
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = nullptr;

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, 1, &kernelParams);

    return kernelNode;
}

cudaGraphNode_t AppendReLUBackwardNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes = {},
    Tensor A, Tensor B, Tensor C) {
    
    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);

    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(A.col, BLOCKSIZE), ceil(A.row, BLOCKSIZE));

    void* kernelArgs[] = { &A.data, &B.data, &C.data, &A.pitch, &B.pitch, &C.pitch, &A.row, &A.col}; 

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)ReLUBackwardKernel;
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = nullptr;

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, 1, &kernelParams);

    return kernelNode;
}