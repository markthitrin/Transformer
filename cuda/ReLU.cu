#include "Header.cuh"
#include "Tensor.cuh"
#include "TensorGraph.cuh"
#include "Util.cuh"
#include "UtilKernel.cuh"
#include "ReLU.cuh"

ReLU::ReLU(
	Tensor& input,
    Tensor& output,
    Tensor& outputGradient,
    Tensor& inputGradient) noexcept :
    input(input),
    output(output),
    outputGradient(outputGradient),
    inputGradient(inputGradient)  { ; }

cudaGraphNode_t ReLU::AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = AppendReLUNode(graph, dependencyNodes, input, output);
    return k1;
}

cudaGraphNode_t ReLU::AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    return AppendGraphForward(graph, dependencyNodes);
}

cudaGraphNode_t ReLU::AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = AppendReLUBackwardNode(graph, dependencyNodes, outputGradient, input, inputGradient);
    return k1;
}



cudaGraphNode_t AppendReLUNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& input, Tensor& output) {
    
    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(input.col, BLOCKSIZE), ceil(input.row, BLOCKSIZE));

    void* kernelArgs[] = { &input.data, &output.data, &input.pitch, &output.pitch, &input.row, &input.col};

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)ReLUKernel;
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = nullptr;

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);

    return kernelNode;
}

cudaGraphNode_t AppendReLUBackwardNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& outputGradient, Tensor& input, Tensor& inputGradient) {
    
    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(outputGradient.col, BLOCKSIZE), ceil(outputGradient.row, BLOCKSIZE));

    void* kernelArgs[] = { 
        &outputGradient.data, &input.data, &inputGradient.data, 
        &outputGradient.pitch, &input.pitch, &inputGradient.pitch, 
        &outputGradient.row, &outputGradient.col}; 

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)ReLUBackwardKernel;
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = nullptr;

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);

    return kernelNode;
}



__global__ void ReLUKernel(
	const float* A, float* C,
	const std::size_t pitchA, const std::size_t pitchC,
	const std::size_t row, const std::size_t col) {

	const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

	if(r < row && c < col) {
		*Get(C, r, c, pitchC) = fmaxf(*Get(A, r, c, pitchA), 0.0f); 
	}
}

__global__ void ReLUBackwardKernel(
	const float* A, const float* B, float* C,
	const std::size_t pitchA, const std::size_t pitchB, const std::size_t pitchC,
	const std::size_t row, const std::size_t col) {
	
	const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

	if(r < row && c < col) {
		*Get(C, r, c, pitchC) = float(*Get(B, r, c, pitchB) > 0) * *Get(A, r, c, pitchA); 
	}
}