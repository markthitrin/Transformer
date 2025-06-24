#ifndef RELU
#define RELU

#include "Header.cuh"
#include "Tensor.cuh"

class ReLU {
public:
	ReLU(
		Tensor& input,
		Tensor& output,
		Tensor& outputGradient,
		Tensor& inputGradient) noexcept;

	cudaGraphNode_t AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	Tensor& input;
	Tensor& output;
	Tensor& outputGradient;
	Tensor& inputGradient;
};


__global__ void ReLUKernel(
	const float* A, float* C,
	const std::size_t pitchA, const std::size_t pitchC,
	const std::size_t row, const std::size_t col);

__global__ void ReLUBackwardKernel(
	const float* A, const float* B, float* C,
	const std::size_t pitchA, const std::size_t pitchB, const std::size_t pitchC,
	const std::size_t row, const std::size_t col);

cudaGraphNode_t AppendReLUNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor input, Tensor output);

cudaGraphNode_t AppendReLUBackwardNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor outputGradient, Tensor input, Tensor inputGradient);


#endif // ! RELU
