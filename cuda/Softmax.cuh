#ifndef SOFTMAX
#define SOFTMAX

#include "Header.cuh"
#include "Tensor.cuh"

class Softmax {
public:
	Softmax(
		Tensor input,
		Tensor output,
		Tensor outputGradient,
		Tensor inputGradient) noexcept;

	cudaGraphNode_t AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphUpdateParameter(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	Tensor& input;
	Tensor& output;
	Tensor& outputGradient;
	Tensor& inputGradient;
};

#endif