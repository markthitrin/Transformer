#ifndef EMBEDDING
#define EMBEDDING

#include "Header.cuh"
#include "Tensor.cuh"
#include "Util.cuh"

class Embedding {
public:
	Embedding(
		std::size_t* input,
		Tensor& output,
		Tensor& outputGradient,
		const std::size_t token) noexcept;
	~Embedding();

	void UpdateGraph(cudaGraphExec_t graphExec);

	cudaGraphNode_t AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphUpdateParameter(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	void loadParam(cnpy::npz_t npFile, std::string prefix);

	void forwardTest(cnpy::npz_t npFile, std::string prefix);

	void checkUpdatedParam(cnpy::npz_t npFile, std::string prefix);

	void backwardTest(cnpy::npz_t npFile, std::string prefix);

	std::size_t* input;
	Tensor& output;
	Tensor& outputGradient;

	std::vector<Tensor> table;
	std::vector<AdamOptimizer> tableOpt;

	std::vector<float*> entries;
	std::vector<cudaGraphNode_t> forwardNodes;
	std::vector<cudaMemcpy3DParms> forwardNodeParams;
	std::vector<cudaGraphNode_t> backwardNodes;
	std::vector<cudaKernelNodeParams> backwardNodeParams;
	std::vector<cudaGraphNode_t> updateParameterNodes;
	std::vector<cudaKernelNodeParams> updateParameterParams;
};

cudaGraphNode_t AppendEmbeddingNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    std::vector<cudaMemcpy3DParms>& forwardNodeParams, std::vector<cudaGraphNode_t>& forwardNodes,
    Tensor& output);
cudaGraphNode_t AppendEmbeddingBackwardNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    std::vector<cudaKernelNodeParams>& backwardNodeParams, std::vector<cudaGraphNode_t>& backwardNodes,
    Tensor& outputGradient);
cudaGraphNode_t AppendEmbeddingUpdateParameterNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    std::vector<cudaKernelNodeParams>& updateParameterParams, std::vector<cudaGraphNode_t>& updateParameterNodes,
    Tensor& outputGradient);

#endif
