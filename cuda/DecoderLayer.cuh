#ifndef DECODER_LAYER
#define DECODER_LAYER

#include "Header.cuh"
#include "Tensor.cuh"
#include "LayerNorm.cuh"
#include "Linear.cuh"
#include "MultiheadAttention.cuh"
#include "DropOut.cuh"
#include "PositionwiseFeedForward.cuh"
#include "Util.cuh"

class DecoderLayer {
public:
	DecoderLayer(
		Tensor& input,
		Tensor& encoderOut,
		Tensor& output,
		Tensor& outputGradient,
		Tensor& inputGradient,
		Tensor& encoderGradient,
		std::size_t*& srcSeqH,
    	std::size_t*& tgtSeqH) noexcept;	

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
	MultiheadAttention mulAtt1;
	DropOut dropout1;

	LayerNorm norm2;
    MultiheadAttention mulAtt2;
    DropOut dropout2;

    LayerNorm norm3;
	PositionwiseFeedForward pff;
	DropOut	dropout3;

	Tensor& input;
	Tensor& encoderOut;
	Tensor& output;
	Tensor& outputGradient;
	Tensor& inputGradient;
	Tensor& encoderGradient;
	std::size_t*& srcSeqH;
	std::size_t*& tgtSeqH;

	Tensor out1;
	Tensor out2;
	Tensor out3;
	Tensor out4;
	Tensor out5;
	Tensor out6;
	Tensor out7;
	Tensor out8;

	Tensor gradient1;
	Tensor gradient2;
	Tensor gradient3;
	Tensor gradient4;
	Tensor gradient5;
	Tensor gradient6;
	Tensor gradient7;
	Tensor gradient8;
};

#endif // !ENCODER_LAYER