#ifndef DECODER
#define DECODER

#include "Header.cuh"
#include "Tensor.cuh"
#include "DecoderLayer.cuh"
#include "Softmax.cuh"
#include "PositionalEncoder.cuh"
#include "Embedding.cuh"
#include "Linear.cuh"
#include "LayerNorm.cuh"
#include "Util.cuh"

class Decoder {
public:
	Decoder(
        Tensor& input,
        Tensor& encoderOut,
		Tensor& output,
		Tensor& outputGradient,
		Tensor& inputGradient,
        Tensor& encoderGradient,
		std::size_t*& srcSeqH,
		std::size_t*& tgtSeqH) noexcept;
	~Decoder();

	void UpdateGraph();

	cudaGraphNode_t AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphUpdateParameter(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

    void loadParam(cnpy::npz_t npFile, std::string prefix);

	void checkUpdatedParam(cnpy::npz_t npFile, std::string prefix);

	void forwardTest(cnpy::npz_t npFile, std::string prefix);

	void backwardTest(cnpy::npz_t npFile, std::string prefix);

    DecoderLayer* layers[N];
	LayerNorm* norm;

	Tensor& input;
	Tensor& encoderOut;
	Tensor& output;
	Tensor& outputGradient;
	Tensor& inputGradient;
	Tensor& encoderGradient;
	std::size_t*& srcSeqH;
	std::size_t*& tgtSeqH;

	std::vector<Tensor> out;

	std::vector<Tensor> gradient;
};

#endif // !CLONE
