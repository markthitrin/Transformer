#ifndef LINEAR
#define LINEAR

#include "Header.cuh"
#include "Tensor.cuh"
#include "Util.cuh"

class Linear {
public:
	Linear(
		Tensor& input,
		Tensor& output,
		Tensor& outputGradient,
		Tensor& inputGradient,
		const std::size_t in,
		const std::size_t out) noexcept;

	cudaGraphNode_t AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphUpdateParameter(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	void loadParam(cnpy::npz_t npFile, std::string prefix);

	void checkUpdatedParam(cnpy::npz_t npFile, std::string prefix);

	Tensor& input;
	Tensor& output;
	Tensor& outputGradient;
	Tensor& inputGradient;

	Tensor weight;
	Tensor bias;

	AdamOptimizer weightOpt;
	AdamOptimizer biasOpt;
};

#endif // !LINEAR
