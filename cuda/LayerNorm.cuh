#ifndef LAYER_NORM
#define LAYER_NORM

#include "Header.cuh"
#include "Tensor.cuh"
#include "TensorFunction.cuh"
#include "Util.cuh"
#include "UtilKernel.cuh"

class LayerNorm {
public:
	LayerNorm(
		Tensor& input,
		Tensor& output,
		Tensor& outputGradient,
		Tensor& inputGradient) noexcept;
	~LayerNorm();

	cudaGraphNode_t AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphUpdateParameter(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	void loadParam(cnpy::npz_t npFile, std::string prefix);

	void checkUpdatedParam(cnpy::npz_t npFile, std::string prefix);

	void forwardTest(cnpy::npz_t npFile, std::string prefix);

	void backwardTest(cnpy::npz_t npFile, std::string prefix);

	Tensor& input;
	Tensor& output;
	Tensor& outputGradient;
	Tensor& inputGradient;

	Tensor alpha;
	Tensor bias;

	AdamOptimizer alphaOpt;
	AdamOptimizer biasOpt;

	Tensor mean;
	Tensor std;
	Tensor xHat;

	Tensor sumG;
	Tensor sumGXhat;
};

cudaGraphNode_t AppendLayerNormNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor input, Tensor mean, Tensor std, Tensor alpha, Tensor bias, Tensor xHat, Tensor output);
cudaGraphNode_t AppendMeanNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor input, Tensor mean);
cudaGraphNode_t AppendStdNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor input, Tensor mean, Tensor std);
cudaGraphNode_t AppendLayerNormBackwardNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor outputGradient, Tensor xHat, Tensor std, Tensor sumG, Tensor sumGXhat, Tensor alpha, Tensor inputGradient);

__global__ void LayerNormKernel(
	const float* A, const float* mean, const float* std, const float* alpha, const float* bias, float* C, float* D,
	const std::size_t pitchA, const std::size_t pitchC, const std::size_t pitchD,
	const std::size_t row, const std::size_t col);
__global__ void LayerNormBackwardKernel(
	const float* A, const float* B, const float* std, const float* sumG, const float* sumGXHat, const float* alpha, float* C,
	const std::size_t pitchA, const std::size_t pitchB, const std::size_t pitchC,
	const std::size_t row, const std::size_t col);


#endif