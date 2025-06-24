#include "Header.cuh"
#include "Tensor.cuh"
#include "TensorFunction.cuh"
#include "Linear.cuh"
#include "ReLU.cuh"
#include "Util.cuh"
#include "DropOut.cuh"
#include "PositionwiseFeedForward.cuh"

PositionwiseFeedForward::PositionwiseFeedForward(
    Tensor input,
    Tensor output,
    Tensor outputGradient,
    Tensor inputGradient) noexcept:

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
    // std::vector<Tensor> updatedTable;
    // for(int i = 0;i < table.size();i++) updatedTable.emplace_back(1, dModel);

    // Tensor loadRR(table.size(), table[0].col);
    // loadRR.loadNp(npFile, prefix + ".updated_weights");
    // for(int i = 0;i < updatedTable.size();i++) {
    //     cudaMemcpy2D(
    //         updatedTable[i].data, updatedTable[i].pitch, Get(loadRR.data, i, 0, loadRR.pitch), loadRR.pitch,
    //         sizeof(float) * updatedTable[i].col, 1, cudaMemcpyDeviceToDevice);
    // }
    // cudaDeviceSynchronize();

    // for(int i = 0;i < updatedTable.size();i++) {
    //     PrintTestResult("backward table:" + std::to_string(i), table[i], updatedTable[i]);
    // }
}

void PositionwiseFeedForward::backwardTest(cnpy::npz_t npFile, std::string prefix) {
    // Set(outputGradient, 1.0f / output.row / output.col);
    // cudaDeviceSynchronize();

    // // load input
    // float* temp = new float[batch * sequenceLength];
    // Tensor loadInput(1, batch * sequenceLength);
    // loadInput.loadNp(npFile, prefix + ".input");
    // loadInput.toFloat(temp);
    // for(int i = 0;i < batch * sequenceLength;i++) {
    //     input[i] = (std::size_t)temp[i];
    // }

    // // Forward, backward, update
    // cudaGraph_t graph;
    // cudaGraphExec_t instance;
    // cudaGraphCreate(&graph, 0);
    // cudaGraphNode_t k1 = this->AppendGraphForward(graph, {});
    // cudaGraphNode_t k2 = this->AppendGraphBackward(graph, {k1});
    // this->AppendGraphUpdateParameter(graph, {k2});
    // cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
    // this->UpdateGraph(instance);
    // cudaGraphLaunch(instance, 0);
    // cudaDeviceSynchronize();

    // checkUpdatedParam(npFile, prefix);
}
