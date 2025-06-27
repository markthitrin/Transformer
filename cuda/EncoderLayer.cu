#include "Header.cuh"
#include "Tensor.cuh"
#include "TensorGraph.cuh"
#include "LayerNorm.cuh"
#include "Linear.cuh"
#include "MultiheadAttention.cuh"
#include "DropOut.cuh"
#include "PositionwiseFeedForward.cuh"
#include "Util.cuh"
#include "EncoderLayer.cuh"


EncoderLayer::EncoderLayer(
    Tensor& input,
    Tensor& output,
    Tensor& outputGradient,
    Tensor& inputGradient,
    std::size_t* srcSeqH) noexcept :

    norm1(input, out1, gradient1, inputGradient),
    mulAtt(out1, out1, out1, out2, gradient2, gradient1, gradient1, gradient1, MaskType::PADDING, srcSeqH),
    dropout1(out2, out3, gradient3, gradient2, batch * sequenceLength, dModel),
    norm2(out3, out4, gradient4, gradient3),
    pff(out4, out5, gradient5, gradient4),
    dropout2(out5, output, outputGradient, gradient5, batch * sequenceLength, dModel),

    input(input),
    output(output),
    outputGradient(outputGradient),
    inputGradient(inputGradient),
    srcSeqH(srcSeqH),

    out1(batch * sequenceLength, dModel),
    out2(batch * sequenceLength, dModel),
    out3(batch * sequenceLength, dModel),
    out4(batch * sequenceLength, dModel),
    out5(batch * sequenceLength, dModel),

    gradient1(batch * sequenceLength, dModel),
    gradient2(batch * sequenceLength, dModel),
    gradient3(batch * sequenceLength, dModel),
    gradient4(batch * sequenceLength, dModel),
    gradient5(batch * sequenceLength, dModel) {;}

void EncoderLayer::UpdateGraph() {
	mulAtt.UpdateGraph();
}

cudaGraphNode_t EncoderLayer::AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = norm1.AppendGraphForward(graph, dependencyNodes);
    cudaGraphNode_t k2 = mulAtt.AppendGraphForward(graph, { k1 });
    cudaGraphNode_t k3 = dropout1.AppendGraphForward(graph, { k2 });
    cudaGraphNode_t k4 = AppendPlusInplaceNode(graph, { k3 }, out3, input);
    cudaGraphNode_t k5 = norm2.AppendGraphForward(graph, { k4 });
    cudaGraphNode_t k6 = pff.AppendGraphForward(graph, { k5 });
    cudaGraphNode_t k7 = dropout2.AppendGraphForward(graph, { k6 });
    cudaGraphNode_t k8 = AppendPlusInplaceNode(graph, { k7 }, output, out3);
    return k8;
}

cudaGraphNode_t EncoderLayer::AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = norm1.AppendGraphPredict(graph, dependencyNodes);
    cudaGraphNode_t k2 = mulAtt.AppendGraphPredict(graph, { k1 });
    cudaGraphNode_t k3 = dropout1.AppendGraphPredict(graph, { k2 });
    cudaGraphNode_t k4 = AppendPlusInplaceNode(graph, { k3 }, out3, input);
    cudaGraphNode_t k5 = norm2.AppendGraphPredict(graph, { k4 });
    cudaGraphNode_t k6 = pff.AppendGraphPredict(graph, { k5 });
    cudaGraphNode_t k7 = dropout2.AppendGraphPredict(graph, { k6 });
    cudaGraphNode_t k8 = AppendPlusInplaceNode(graph, { k7 }, output, out3);
    return k8;
}

cudaGraphNode_t EncoderLayer::AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = dropout2.AppendGraphBackward(graph, dependencyNodes);
    cudaGraphNode_t k2 = pff.AppendGraphBackward(graph, { k1 });
    cudaGraphNode_t k3 = norm2.AppendGraphBackward(graph, { k2 });
    cudaGraphNode_t k4 = AppendPlusInplaceNode(graph, { k3 }, gradient3, outputGradient);
    cudaGraphNode_t k5 = dropout1.AppendGraphBackward(graph, { k4 });
    cudaGraphNode_t k6 = mulAtt.AppendGraphBackward(graph, { k5 });
    cudaGraphNode_t k7 = norm1.AppendGraphBackward(graph, { k6 });
    cudaGraphNode_t k8 = AppendPlusInplaceNode(graph, { k7 }, inputGradient, gradient3);
    return k8;
}

cudaGraphNode_t EncoderLayer::AppendGraphUpdateParameter(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = SyncDependency(graph, dependencyNodes);
    cudaGraphNode_t k2 = norm1.AppendGraphUpdateParameter(graph, { k1 });
    cudaGraphNode_t k3 = mulAtt.AppendGraphUpdateParameter(graph, { k1 });
    cudaGraphNode_t k4 = norm2.AppendGraphUpdateParameter(graph, { k1 });
    cudaGraphNode_t k5 = pff.AppendGraphUpdateParameter(graph, { k1 });
    cudaGraphNode_t k6 = SyncDependency(graph, { k2, k3, k4, k5 });
    return k6;
}

void EncoderLayer::loadParam(cnpy::npz_t npFile, std::string prefix) {
    norm1.loadParam(npFile, prefix + ".sub1.layerNorm");
    mulAtt.loadParam(npFile, prefix + ".sub1.sublayer");
    norm2.loadParam(npFile, prefix + ".sub2.layerNorm");
    pff.loadParam(npFile,prefix + ".sub2.sublayer");
}

void EncoderLayer::checkUpdatedParam(cnpy::npz_t npFile, std::string prefix) {
    norm1.checkUpdatedParam(npFile, prefix + ".sub1.layerNorm");
    mulAtt.checkUpdatedParam(npFile, prefix + ".sub1.sublayer");
    norm2.checkUpdatedParam(npFile, prefix + ".sub2.layerNorm");
    pff.checkUpdatedParam(npFile,prefix + ".sub2.sublayer");
}

void EncoderLayer::forwardTest(cnpy::npz_t npFile, std::string prefix) {
    Tensor target(output.row, output.col);

    target.loadNp(npFile, prefix + ".output");
    input.loadNp(npFile, prefix + ".input");
	Tensor npdLoader(1, 1);
	npdLoader.loadNp(npFile, prefix + ".npd");
	float* _seqH = new float[batch];
	npdLoader.toFloat((float*)_seqH);
	for(int i = 0;i < batch;i++) {
		srcSeqH[i] = _seqH[0];
	}

    // Forward
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaGraphCreate(&graph, 0);
    this->AppendGraphForward(graph, {});
    cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
	this->UpdateGraph();
    cudaGraphLaunch(instance, 0);
    cudaDeviceSynchronize();

    PrintTestResult("forward", output, target);
}

void EncoderLayer::backwardTest(cnpy::npz_t npFile, std::string prefix) {
    Set(outputGradient, 1.0f / output.row / output.col);
    cudaDeviceSynchronize();

    // load input
    input.loadNp(npFile, prefix + ".input");
    Tensor npdLoader(1, 1);
	npdLoader.loadNp(npFile, prefix + ".npd");
	float* _seqH = new float[batch];
	npdLoader.toFloat((float*)_seqH);
	for(int i = 0;i < batch;i++) {
		srcSeqH[i] = _seqH[0];
	}

    // Forward, backward, update
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaGraphCreate(&graph, 0);
    cudaGraphNode_t k1 = this->AppendGraphForward(graph, {});
    cudaGraphNode_t k2 = this->AppendGraphBackward(graph, {k1});
    this->AppendGraphUpdateParameter(graph, {k2});
    cudaGraphDebugDotPrint(graph, "graph.dot", cudaGraphDebugDotFlagsVerbose);
    cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
    this->UpdateGraph();
    cudaGraphLaunch(instance, 0);
    cudaDeviceSynchronize();
    // Print(gradient1, 0, 0, 10, 10);
    // Print(gradient2, 0, 0, 10, 10);
    // Print(gradient3, 0, 0, 10, 10);
    // Print(linear1.biasOpt.gradient, 0, 0, 10 ,10);
    
    checkUpdatedParam(npFile, prefix);
}