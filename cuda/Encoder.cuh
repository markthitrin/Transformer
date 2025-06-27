#ifndef ENCODER
#define ENCODER

#include "Header.cuh"
#include "Tensor.cuh"
#include "EncoderLayer.cuh"
#include "Softmax.cuh"
#include "PositionalEncoder.cuh"
#include "Embedding.cuh"
#include "Linear.cuh"
#include "LayerNorm.cuh"
#include "Util.cuh"

class Encoder {
public:
	Encoder(
		Tensor& input,
		Tensor& output,
		Tensor& outputGradient,
		Tensor& inputGradient,
		std::size_t* srcSeqH) noexcept;
	~Encoder() noexcept;

	void UpdateGraph();

	cudaGraphNode_t AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphUpdateParameter(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

    void loadParam(cnpy::npz_t npFile, std::string prefix);

	void checkUpdatedParam(cnpy::npz_t npFile, std::string prefix);

	void forwardTest(cnpy::npz_t npFile, std::string prefix);

	void backwardTest(cnpy::npz_t npFile, std::string prefix);

	EncoderLayer* layers[N]; // using pointer as copy anad move constructor isn't implemented.
	LayerNorm* norm;

	Tensor& input;
	Tensor& output;
	Tensor& outputGradient;
    Tensor& inputGradient;
	std::size_t* srcSeqH;

	std::vector<Tensor> out;
	
	std::vector<Tensor> gradient;
};

#endif 
