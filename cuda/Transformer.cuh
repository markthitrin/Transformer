#ifndef TRANSFORMER
#define TRANSFORMER

#include "Header.cuh"
#include "Tensor.cuh"
#include "Encoder.cuh"
#include "Softmax.cuh"
#include "PositionalEncoder.cuh"
#include "Embedding.cuh"
#include "Linear.cuh"
#include "LayerNorm.cuh"
#include "Util.cuh"
#include "Decoder.cuh"

// Important~~~ Currently memory are leaking from build a graph node with single float as parameter
class Transformer {
public:
	Transformer() noexcept;
    ~Transformer();

    void UpdateGraph(cudaGraphExec_t instance);

    cudaGraphNode_t AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphUpdateParameter(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

    void loadParam(cnpy::npz_t npFile, std::string prefix);

	void checkUpdatedParam(cnpy::npz_t npFile, std::string prefix);

	void forwardTest(cnpy::npz_t npFile, std::string prefix);

	void backwardTest(cnpy::npz_t npFile, std::string prefix);

    Embedding srcEmbed;
    Embedding tgtEmbed;
    PositionalEncoder srcPos;
    PositionalEncoder tgtPos;
    Decoder decoder;
    Encoder encoder;
    Linear linear;

	std::size_t* inputEncoderH;
    std::size_t* inputDecoderH;
    std::size_t* srcSeqH;
    std::size_t* tgtSeqH;
	Tensor output;
	Tensor outputGradient;

	Tensor encoderOut;
    Tensor encoderGradient;
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

#endif // !TRANSFORMER
