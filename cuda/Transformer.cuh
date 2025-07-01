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

    float Train(const std::size_t* encoderInput, const std::size_t* srcSeq,
        const std::size_t* decoderInput, const std::size_t* tgtSeq,
        const std::size_t* tragetOutput);

    void Encode(const std::size_t* encoderInput, const std::size_t* srcSeq);

    void Decode(const std::size_t* decoderInput, const std::size_t* tgtSeq);

    void ResetGraph();

    void SetTrainGraph();

    void SetPredictGraph();

    void UpdateGraph(cudaGraphExec_t instance);

    cudaGraphNode_t AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphPredictEncode(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);
	cudaGraphNode_t AppendGraphPredictDecode(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphUpdateParameter(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

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

    cudaGraph_t graphForward;
    cudaGraphExec_t graphExecForward;
    cudaGraph_t graphBackward;
    cudaGraphExec_t graphExecBackward;
    cudaGraph_t graphEncode;
    cudaGraphExec_t graphExecEncode;
    cudaGraph_t graphDecode;
    cudaGraphExec_t graphExecDecode;

    int graphState = 0;
};

#endif // !TRANSFORMER
