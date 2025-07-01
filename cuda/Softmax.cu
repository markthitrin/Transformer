#include "Header.cuh"
#include "Tensor.cuh"
#include "TensorGraph.cuh"
#include "Util.cuh"
#include "UtilKernel.cuh"
#include "Softmax.cuh"

Softmax::Softmax(
    Tensor& input,
    Tensor& output,
    Tensor& outputGradient,
    Tensor& inputGradient,
	const std::size_t row,
	const std::size_t col) noexcept:

    input(input),
    output(output),
    outputGradient(outputGradient),
    inputGradient(inputGradient),
	
	maxValue(1, row),
	sumExp(1, row),
	sumGY(1, row) { ; }

cudaGraphNode_t Softmax::AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
	cudaGraphNode_t k1 = AppendReduceMaxNode(graph, dependencyNodes, input, maxValue);
	cudaGraphNode_t k2 = AppendReduceSumExpNode(graph, {k1}, input, maxValue, sumExp);
	cudaGraphNode_t k3 = AppendSoftmaxNode(graph, {k2}, input, maxValue, sumExp, output);
	return k3;
}

cudaGraphNode_t Softmax::AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
	return AppendGraphForward(graph, dependencyNodes);
}

cudaGraphNode_t Softmax::AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
	cudaGraphNode_t k1 = AppendReduceSumOfProductNode(graph, dependencyNodes, output, outputGradient, sumGY);
	cudaGraphNode_t k2 = AppendSoftmaxBackwardNode(graph, {k1}, output, outputGradient, sumGY, inputGradient);
	return k2;
}



cudaGraphNode_t AppendSoftmaxNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& input, Tensor& maxValue, Tensor& sumExp, Tensor& output) {
    
    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

	const int BLOCK_SIZE = 16;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(ceil(input.col, BLOCK_SIZE), ceil(input.row, BLOCK_SIZE));

    void* kernelArgs[] = {
		&input.data, &maxValue.data, &sumExp.data, &output.data,
		&input.pitch, &output.pitch,
		&input.row, &input.col};

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)SoftmaxKernel;
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = nullptr;

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);

    return kernelNode;
}

cudaGraphNode_t AppendSoftmaxBackwardNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& output, Tensor& outputGradient, Tensor& sumGY, Tensor& inputGradient) {
    
    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

	const int BLOCK_SIZE = 16;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(ceil(output.col, BLOCK_SIZE), ceil(output.row, BLOCK_SIZE));

    void* kernelArgs[] = {
		&output.data, &outputGradient, &sumGY, &inputGradient,
		&output.pitch, &outputGradient.pitch, &inputGradient.pitch,
		&output.row, &output.col};

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)SoftmaxBackwardKernel;
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = nullptr;

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);

    return kernelNode;
}



__global__ void SoftmaxKernel (
	const float* input, const float* maxValue, const float* sumExp, float* output,
	const std::size_t pitchInput, const std::size_t pitchOutput,
	const std::size_t row, const std::size_t col) {

	const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

	if(r < row && c < col) {
		*Get(output, r, c, pitchOutput) = expf(*Get(input, r, c, pitchInput) - maxValue[r]) / (sumExp[r] + eps);
	}
}

__global__ void SoftmaxBackwardKernel(
	const float* output, const float* outputGradient, const float* sumGY, float* inputGradient,
	const std::size_t pitchOutput, const std::size_t pitchOutputGradient, const std::size_t pitchInputGradient,
	const std::size_t row, const std::size_t col) {

	const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(r < row && c < col) {
		*Get(inputGradient, r, c, pitchInputGradient) = *Get(output, r, c, pitchOutput) * (*Get(outputGradient, r, c, pitchOutputGradient) - sumGY[r]);
	}
}