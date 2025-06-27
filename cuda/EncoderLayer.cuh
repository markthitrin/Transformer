#ifndef ENCODER_LAYER
#define ENCODER_LAYER

#include "Header.cuh"
#include "Tensor.cuh"
#include "LayerNorm.cuh"
#include "Linear.cuh"
#include "MultiheadAttention.cuh"
#include "DropOut.cuh"
#include "PositionwiseFeedForward.cuh"
#include "Util.cuh"

class EncoderLayer {
public:
	EncoderLayer(
		Tensor& input,
		Tensor& output,
		Tensor& outputGradient,
		Tensor& inputGradient,
		std::size_t* srcSeqH) noexcept;

	void UpdateGraph();

	cudaGraphNode_t AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphUpdateParameter(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	void loadParam(cnpy::npz_t npFile, std::string prefix);

	void checkUpdatedParam(cnpy::npz_t npFile, std::string prefix);

	void forwardTest(cnpy::npz_t npFile, std::string prefix);

	void backwardTest(cnpy::npz_t npFile, std::string prefix);

	LayerNorm norm1;
	MultiheadAttention mulAtt;
	DropOut dropout1;
	LayerNorm norm2;
	PositionwiseFeedForward pff;
	DropOut	dropout2;

	Tensor& input;
	Tensor& output;
	Tensor& outputGradient;
	Tensor& inputGradient;
	std::size_t* srcSeqH;

	Tensor out1;
	Tensor out2;
	Tensor out3;
	Tensor out4;
	Tensor out5;

	Tensor gradient1;
	Tensor gradient2;
	Tensor gradient3;
	Tensor gradient4;
	Tensor gradient5;
};

#endif // !ENCODER_LAYER