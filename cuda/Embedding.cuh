#ifndef EMBEDDING
#define EMBEDDING

#include "Header.cuh"
#include "Tensor.cuh"
#include "Util.cuh"

class Embedding {
public:
	Embedding(
		std::size_t*& inputH,
		Tensor& output,
		Tensor& outputGradient,
		const std::size_t numToken) noexcept;
	~Embedding();

	void UpdateGraph(cudaGraphExec_t instance);

	cudaGraphNode_t AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphUpdateParameter(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	std::size_t*& inputH;
	std::size_t* input;
	Tensor& output;
	Tensor& outputGradient;
	const std::size_t numToken;

	Tensor table;
	AdamOptimizer tableOpt;
	std::size_t* t;

	std::vector<cudaGraphNode_t> forwardNodes;
	std::vector<cudaMemcpy3DParms> forwardParams;
};

cudaGraphNode_t AppendEmbeddingNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    std::size_t*& input, std::vector<cudaGraphNode_t>& forwardNodes, std::vector<cudaMemcpy3DParms>& forwardParams,
    Tensor& table, Tensor& output);
cudaGraphNode_t AppendEmbeddingBackwardNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    std::size_t*& input,
    AdamOptimizer& tableOpt, Tensor& outputGradient);
cudaGraphNode_t AppendEmbeddingUpdateParameterNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    std::size_t* input,
    Tensor& table, AdamOptimizer& tableOpt, std::size_t*& t, Tensor& outputGradient);

__global__ void EmbeddingBackwardKernel(
    const float* outputGradient, const std::size_t* input, float* gradient,
	const std::size_t pitchGradient,
    const std::size_t col);
__global__ void EmbeddingAdamOptKernel(
    const std::size_t* input, float* param, float* gradient, float* accM, float* accV, std::size_t* t,
    const std::size_t pitchParam, const std::size_t pitchGrad, const std::size_t pitchAccM, const std::size_t pitchAccV,
    const std::size_t row, const std::size_t col);

#endif
