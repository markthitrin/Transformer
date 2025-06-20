#ifndef LINEAR
#define LINEAR

#include "Header.h"
#include "Tensor.cuh"
#include "Util.cuh"

class Linear {
public:
	Linear(
		Tensor input,
		Tensor output,
		Tensor outputGradient,
		Tensor inputGradient,
		const std::size_t in,
		const std::size_t out) noexcept;
	~Linear();

	cudaGraphNode_t AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes = {});

	cudaGraphNode_t AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes = {});

	cudaGraphNode_t AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes = {});

	cudaGraphNode_t AppendGraphUpdateParameter(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes = {});

	Tensor& input;
	Tensor& output;
	Tensor& outputGradient;
	Tensor& inputGradient;

	Tensor weight;
	Tensor bias;

	int feedCount = 0;
	AdamOptimizer weightOpt;
	AdamOptimizer biasOpt;
};

#endif // !LINEAR
