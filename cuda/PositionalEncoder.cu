#include "PositionalEncoder.cuh"
#include "TensorFunction.cuh"
#include "TensorGraph.cuh"
#include "Util.cuh"

PositionalEncoder::PositionalEncoder(
    Tensor& input,
    Tensor& output,
    Tensor& outputGradient,
    Tensor& inputGradient) noexcept :

    dropout(input, output, outputGradient, inputGradient, batch * sequenceLength, dModel),

    input(input),
    output(output),
    outputGradient(outputGradient),
    inputGradient(inputGradient),
    positionEncode(sequenceLength, dModel) {

    GetPositionalEncode(positionEncode);
}

cudaGraphNode_t PositionalEncoder::AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = AppendPlusInplaceBatchNode(graph, dependencyNodes, input, positionEncode, batch);
    cudaGraphNode_t k2 = dropout.AppendGraphForward(graph, {k1});
    return k2;
}

cudaGraphNode_t PositionalEncoder::AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    return AppendGraphForward(graph, dependencyNodes);
}

cudaGraphNode_t PositionalEncoder::AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = dropout.AppendGraphBackward(graph, dependencyNodes);
    return k1;
}
