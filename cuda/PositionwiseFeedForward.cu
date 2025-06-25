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

PositionwiseFeedForward::~PositionwiseFeedForward() noexcept {
    out1.free();
    out2.free();
    out3.free();
    gradient1.free();
    gradient2.free();
    gradient3.free();
}

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

void PositionwiseFeedForward::loadParam(cnpy::npz_t npFile, std::string prefix) {
    linear1.weight.loadNp(npFile, prefix + ".w1");
    linear1.bias.loadNp(npFile, prefix + ".b1");
    linear2.weight.loadNp(npFile, prefix + ".w2");
    linear2.bias.loadNp(npFile, prefix + ".b2");
}

void PositionwiseFeedForward::forwardTest(cnpy::npz_t npFile, std::string prefix) {
    Tensor target(output.row, output.col);

    target.loadNp(npFile, prefix + ".output");
    input.loadNp(npFile, prefix + ".input");

    // Forward
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaGraphCreate(&graph, 0);
    this->AppendGraphForward(graph, {});
    cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
    cudaGraphLaunch(instance, 0);
    cudaDeviceSynchronize();

    PrintTestResult("forward", output, target);
}

void PositionwiseFeedForward::checkUpdatedParam(cnpy::npz_t npFile, std::string prefix) {
    Tensor updatedW1(dModel, dFF);
    Tensor updatedB1(1, dFF);
    Tensor updatedW2(dFF, dModel);
    Tensor updatedB2(1, dModel);
    updatedW1.loadNp(npFile, prefix + ".updated_w1");
    updatedB1.loadNp(npFile, prefix + ".updated_b1");
    updatedW2.loadNp(npFile, prefix + ".updated_w2");
    updatedB2.loadNp(npFile, prefix + ".updated_b2");

    PrintTestResult("backward [" + prefix + ".updated_w1" + "]", linear1.weight,updatedW1);
    PrintTestResult("backward [" + prefix + ".updated_b1" + "]", linear1.bias,updatedB1);
    PrintTestResult("backward [" + prefix + ".updated_w2" + "]", linear2.weight,updatedW2);
    PrintTestResult("backward [" + prefix + ".updated_b2" + "]", linear2.bias,updatedB2);
}

void PositionwiseFeedForward::backwardTest(cnpy::npz_t npFile, std::string prefix) {
    Set(outputGradient, 1.0f / output.row / output.col);
    cudaDeviceSynchronize();

    // load input
    input.loadNp(npFile, prefix + ".input");

    // Forward, backward, update
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaGraphCreate(&graph, 0);
    cudaGraphNode_t k1 = this->AppendGraphForward(graph, {});
    cudaGraphNode_t k2 = this->AppendGraphBackward(graph, {k1});
    this->AppendGraphUpdateParameter(graph, {k2});
    cudaGraphDebugDotPrint(graph, "graph.dot", cudaGraphDebugDotFlagsVerbose);
    cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
    cudaGraphLaunch(instance, 0);
    cudaDeviceSynchronize();
    // Print(gradient1, 0, 0, 10, 10);
    // Print(gradient2, 0, 0, 10, 10);
    // Print(gradient3, 0, 0, 10, 10);
    // Print(linear1.biasOpt.gradient, 0, 0, 10 ,10);
    
    checkUpdatedParam(npFile, prefix);
}
