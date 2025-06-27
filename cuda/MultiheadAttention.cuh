#ifndef MULTIHEAD_ATTENTION
#define MULTIHEAD_ATTENTION

#include "Header.cuh"
#include "Tensor.cuh"
#include "Softmax.cuh"
#include "DropOut.cuh"
#include "Util.cuh"

enum MaskType {
	LOOK_AHEAD,
	PADDING,
	CROSS_PADDING
};

class MultiheadAttention {
public:
	MultiheadAttention(
		Tensor& inputQ,
		Tensor& inputK,
		Tensor& inputV,
		Tensor& output,
		Tensor& outputGradient,
		Tensor& inputGradientQ,
		Tensor& inputGradientK,
		Tensor& inputGradientV,
		MaskType maskType,
		std::size_t* seqH) noexcept;
	~MultiheadAttention();

	void UpdateGraph(cudaGraphExec_t graphExec);

	cudaGraphNode_t AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphUpdateParameter(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	void loadParam(cnpy::npz_t npFile, std::string prefix);

	void forwardTest(cnpy::npz_t npFile, std::string prefix);

	void checkUpdatedParam(cnpy::npz_t npFile, std::string prefix);

	void backwardTest(cnpy::npz_t npFile, std::string prefix);

	Softmax softmax;
	DropOut dropout;

	Tensor& inputQ;
	Tensor& inputK;
	Tensor& inputV;
	Tensor& output;
	Tensor& outputGradient;
	Tensor& inputGradientQ;
	Tensor& inputGradientK;
	Tensor& inputGradientV;
	MaskType maskType;
	std::size_t* seqH;
	std::size_t* seq;

	Tensor WQ;
	Tensor WK;
	Tensor WV;
	Tensor WO;

	AdamOptimizer WQOpt;
	AdamOptimizer WKOpt;
	AdamOptimizer WVOpt;
	AdamOptimizer WOOpt;

	Tensor QT;
	Tensor KT;
	Tensor VT;
	Tensor A;
	Tensor As;
	Tensor Ad;
	Tensor OT;

	Tensor QTGradient;
	Tensor KTGradient;
	Tensor VTGradient;
	Tensor AGradient;
	Tensor AsGradient;
	Tensor AdGradient;
	Tensor OTGradient;

	cudaGraphNode_t forwardMaskNode;
	cudaGraphNodeParams forwardMaskParams;
	cudaGraphNode_t backwardMaskNode;
	cudaGraphNodeParams backwardMaskParams;
};

#endif
