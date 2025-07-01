#ifndef POSITIONWISE_FEED_FORWARD
#define POSITIONWISE_FEED_FORWARD

#include "Header.cuh"
#include "Tensor.cuh"
#include "Linear.cuh"
#include "ReLU.cuh"
#include "DropOut.cuh"

class PositionwiseFeedForward {
public:
	PositionwiseFeedForward(
		Tensor& input,
		Tensor& output,
		Tensor& outputGradient,
		Tensor& inputGradient) noexcept;

	cudaGraphNode_t AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphUpdateParameter(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	Linear linear1;
	ReLU relu;
	DropOut dropout;
	Linear linear2;

	Tensor& input;
	Tensor& output;
	Tensor& outputGradient;
	Tensor& inputGradient;

	Tensor out1;
	Tensor out2;
	Tensor out3;

	Tensor gradient1;
	Tensor gradient2;
	Tensor gradient3;
};

#endif