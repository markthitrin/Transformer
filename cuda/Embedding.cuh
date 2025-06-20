#ifndef EMBEDDING
#define EMBEDDING

#include "Header.cuh"
#include "Tensor.cuh"
#include "Util.cuh"

class Embedding {
public:
	Embedding(
		Tensor input,
		Tensor output,
		Tensor outputGradient,
		const std::size_t token) noexcept;
	~Embedding();

	cudaGraphNode_t AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes = {});
	void UpdateGraphForward(cudaGraphExec_t graphExec);

	cudaGraphNode_t AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes = {});

	cudaGraphNode_t AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes = {});

	cudaGraphNode_t AppendGraphUpdateParameter(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes = {});

	Tensor input;
	Tensor output;
	Tensor outputGradient;

	std::vector<int> feedCount;
	std::vector<Tensor> table;
	std::vector<AdamOptimizer> tableOpt;

	cudaGraphNode_t node;
};

#endif
