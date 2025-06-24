#ifndef POSITIONAL_ENCODER
#define POSITIONAL_ENCODER

#include "Header.cuh"
#include "Tensor.cuh"
#include "DropOut.cuh"

class PositionalEncoder {
public:
	PositionalEncoder(
		Tensor& input,
		Tensor& output,
		Tensor& outputGradient,
		Tensor& inputGradient) noexcept;

	cudaGraphNode_t AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	void forwardTest(cnpy::npz_t npFile, std::string prefix);

	DropOut dropout;
	
	Tensor& input;
	Tensor& output;
	Tensor& outputGradient;
	Tensor& inputGradient;

	Tensor positionEncode;
};

#endif
