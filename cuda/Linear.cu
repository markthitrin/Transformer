#include "Header.cuh"
#include "Linear.cuh"
#include "TensorGraph.cuh"

Linear::Linear(
    Tensor& input,
    Tensor& output,
    Tensor& outputGradient,
    Tensor& inputGradient,
    const std::size_t in,
    const std::size_t out) noexcept:

    input(input),
    output(output),
    outputGradient(outputGradient),
    inputGradient(inputGradient),

    weight(in, out),
    bias(1, out),

    weightOpt(in, out),
    biasOpt(1, out) {

    weight.HeNormalFill();
    bias.HeNormalFill();
}
Linear::~Linear() {
    weight.free();
    bias.free();
}

cudaGraphNode_t Linear::AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = AppendCopyBatchNode(graph, dependencyNodes, bias, output, batch * sequenceLength);
    cudaGraphNode_t k2 = AppendMatMulPlusNode(graph, {k1}, input, weight, output, false, false);
    return k2;
}

cudaGraphNode_t Linear::AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    return AppendGraphForward(graph, dependencyNodes);
}

cudaGraphNode_t Linear::AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    feedCount++;
    cudaGraphNode_t k1 = AppendResetNode(graph, dependencyNodes, inputGradient);
    cudaGraphNode_t k2 = AppendPlusInplceBatchNode(graph, {k1}, outputGradient, biasOpt.gradient, outputGradient.row);
    cudaGraphNode_t k3 = AppendMatMulPlusNode(graph, {k2}, outputGradient, input, weightOpt.gradient, true, false);
    cudaGraphNode_t k4 = AppendMatMulPlusNode(graph, {k2}, outputGradient, weight, inputGradient, false, false);
    cudaGraphNode_t k5 = SyncDependency(graph, {k3, k4});
    return k5;
}

cudaGraphNode_t Linear::AppendGraphUpdateParameter(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = SyncDependency(graph, dependencyNodes);
    cudaGraphNode_t k2 = AppendAdamOptNode(graph, {k1}, weight, weightOpt, feedCount);
    cudaGraphNode_t k3 = AppendAdamOptNode(graph, {k1}, bias, biasOpt, feedCount);
    cudaGraphNode_t k4 = SyncDependency(graph, {k2, k3});
    return k4;
}

void Linear::loadParam(cnpy::npz_t npFile, std::string prefix) {
    weight.loadNp(npFile, prefix + ".weight");
    bias.loadNp(npFile, prefix + ".bias");
}