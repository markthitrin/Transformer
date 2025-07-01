#include "Header.cuh"
#include "Tensor.cuh"
#include "TensorFunction.cuh"
#include "Linear.cuh"
#include "ReLU.cuh"
#include "Util.cuh"
#include "DropOut.cuh"
#include "PositionwiseFeedForward.cuh"

PositionwiseFeedForward::PositionwiseFeedForward(
    Tensor& input,
    Tensor& output,
    Tensor& outputGradient,
    Tensor& inputGradient) noexcept:

    linear1(input, out1, gradient1, inputGradient, dModel, dFF),
    relu(out1, out2, gradient2, gradient1),
    dropout(out2, out3, gradient3, gradient2, batch * sequenceLength, dFF),
    linear2(out3, output, outputGradient, gradient3, dFF, dModel),

    input(input),
    output(output),
    outputGradient(outputGradient),
    inputGradient(inputGradient),
    
    out1(batch * sequenceLength, dFF),
    out2(batch * sequenceLength, dFF),
    out3(batch * sequenceLength, dFF),

    gradient1(batch * sequenceLength, dFF),
    gradient2(batch * sequenceLength, dFF),
    gradient3(batch * sequenceLength, dFF) { ; }

cudaGraphNode_t PositionwiseFeedForward::AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = linear1.AppendGraphForward(graph, dependencyNodes);
    cudaGraphNode_t k2 = relu.AppendGraphForward(graph, {k1});
    cudaGraphNode_t k3 = dropout.AppendGraphForward(graph, {k2});
    cudaGraphNode_t k4 = linear2.AppendGraphForward(graph, {k3});
    return k4;
}

cudaGraphNode_t PositionwiseFeedForward::AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = linear1.AppendGraphPredict(graph, dependencyNodes);
    cudaGraphNode_t k2 = relu.AppendGraphPredict(graph, {k1});
    cudaGraphNode_t k3 = dropout.AppendGraphPredict(graph, {k2});
    cudaGraphNode_t k4 = linear2.AppendGraphPredict(graph, {k3});
    return k4;
}

cudaGraphNode_t PositionwiseFeedForward::AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = linear2.AppendGraphBackward(graph, dependencyNodes);
    cudaGraphNode_t k2 = dropout.AppendGraphBackward(graph, {k1});
    cudaGraphNode_t k3 = relu.AppendGraphBackward(graph, {k2});
    cudaGraphNode_t k4 = linear1.AppendGraphBackward(graph, {k3});
    return k4;
}

cudaGraphNode_t PositionwiseFeedForward::AppendGraphUpdateParameter(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = linear1.AppendGraphUpdateParameter(graph, dependencyNodes);
    cudaGraphNode_t k2 = linear2.AppendGraphUpdateParameter(graph, dependencyNodes);
    cudaGraphNode_t k3 = SyncDependency(graph, {k1,k2});
    return k3;
}
