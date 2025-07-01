#include "Header.cuh"
#include "Tensor.cuh"
#include "DecoderLayer.cuh"
#include "Softmax.cuh"
#include "PositionalEncoder.cuh"
#include "Embedding.cuh"
#include "Linear.cuh"
#include "LayerNorm.cuh"
#include "Util.cuh"
#include "Decoder.cuh"

Decoder::Decoder(
    Tensor& input,
    Tensor& encoderOut,
    Tensor& output,
    Tensor& outputGradient,
    Tensor& inputGradient,
    Tensor& encoderGradient,
    std::size_t*& srcSeqH,
	std::size_t*& tgtSeqH) noexcept :

    input(input),
    encoderOut(encoderOut),
    output(output),
    outputGradient(outputGradient),
    inputGradient(inputGradient),
    encoderGradient(encoderGradient),
    
    srcSeqH(srcSeqH),
    tgtSeqH(tgtSeqH) {
    
    out.reserve(N);
    gradient.reserve(N);
    for(int i = 0;i < N;i++) {
        out.emplace_back(batch * sequenceLength, dModel);
        gradient.emplace_back(batch * sequenceLength, dModel);
    }

    norm = new LayerNorm(out[N - 1], output, outputGradient, gradient[N - 1]);
    layers[0] = new DecoderLayer(input, encoderOut, out[0], gradient[0], inputGradient, encoderGradient, srcSeqH, tgtSeqH);
    for(int i = 1;i < N;i++) {
        layers[i] = new DecoderLayer(out[i - 1], encoderOut, out[i], gradient[i], gradient[i - 1], encoderGradient, srcSeqH, tgtSeqH);
    }
}
Decoder::~Decoder() {
    delete norm;
    for(int i = 0;i < N;i++) {
        delete layers[i];
    }
}

void Decoder::UpdateGraph() {
    for(int i = 0;i < N;i++) {
        layers[i]->UpdateGraph();
    }
}

cudaGraphNode_t Decoder::AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1[N];
    k1[0] = layers[0]->AppendGraphForward(graph, dependencyNodes);
    for(int i = 1;i < N;i++) {
        k1[i] = layers[i]->AppendGraphForward(graph, { k1[i - 1] });
    }
    cudaGraphNode_t k2 = norm->AppendGraphForward(graph, { k1[N - 1] });
    return k2;
}

cudaGraphNode_t Decoder::AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1[N];
    k1[0] = layers[0]->AppendGraphPredict(graph, dependencyNodes);
    for(int i = 1;i < N;i++) {
        k1[i] = layers[i]->AppendGraphPredict(graph, { k1[i - 1] });
    }
    cudaGraphNode_t k2 = norm->AppendGraphPredict(graph, { k1[N - 1] });
    return k2;
}

cudaGraphNode_t Decoder::AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = norm->AppendGraphBackward(graph, dependencyNodes);
    cudaGraphNode_t k2[N];
    k2[N - 1] = layers[N - 1]->AppendGraphBackward(graph, { k1 });
    for(int i = N - 2;i >= 0;i--) {
        k2[i] = layers[i]->AppendGraphBackward(graph, { k2[i + 1] });
    }
    return k2[0];
}

cudaGraphNode_t Decoder::AppendGraphUpdateParameter(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = SyncDependency(graph, dependencyNodes);
    cudaGraphNode_t k2 = norm->AppendGraphUpdateParameter(graph, {k1});
    std::vector<cudaGraphNode_t> k3(N);
    for(int i = 0;i < N;i ++) {
        k3[i] = layers[i]->AppendGraphUpdateParameter(graph, {k1});
    }
    cudaGraphNode_t k4 = SyncDependency(graph, k3);
    cudaGraphNode_t k5 = SyncDependency(graph, { k2, k4 });
    return k5;
}