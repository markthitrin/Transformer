#ifndef SOFTMAX
#define SOFTMAX

#include "Header.cuh"
#include "Tensor.cuh"

class Softmax {
public:
	Softmax(
		Tensor& input,
		Tensor& output,
		Tensor& outputGradient,
		Tensor& inputGradient,
		const std::size_t row,
		const std::size_t col) noexcept;

	cudaGraphNode_t AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphUpdateParameter(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	Tensor& input;
	Tensor& output;
	Tensor& outputGradient;
	Tensor& inputGradient;

	Tensor maxValue;
	Tensor sumExp;
	Tensor sumGY;
};

cudaGraphNode_t AppendSoftmaxNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& input, Tensor& maxValue, Tensor& sumExp, Tensor& output);
cudaGraphNode_t AppendSoftmaxBackwardNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& output, Tensor& outputGradient, Tensor& sumGY, Tensor& inputGradient);

__global__ void SoftmaxKernel (
	const float* input, const float* maxValue, const float* sumExp, float* output,
	const std::size_t pitchInput, const std::size_t pitchOutput,
	const std::size_t row, const std::size_t col);
__global__ void SoftmaxBackwardKernel(
	const float* output, const float* outputGradient, const float* sumGY, float* inputGradient,
	const std::size_t pitchOutput, const std::size_t pitchOutputGradient, const std::size_t pitchInputGradient,
	const std::size_t row, const std::size_t col);

#endif