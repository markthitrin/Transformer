#include "Header.cuh"
#include "Tensor.cuh"
#include "TensorGraph.cuh"
#include "LayerNorm.cuh"
#include "Linear.cuh"
#include "MultiheadAttention.cuh"
#include "DropOut.cuh"
#include "PositionwiseFeedForward.cuh"
#include "Util.cuh"
#include "DecoderLayer.cuh"

DecoderLayer::DecoderLayer(
    Tensor& input,
    Tensor& encoderOut,
    Tensor& output,
    Tensor& outputGradient,
    Tensor& inputGradient,
    Tensor& encoderGradient,
    std::size_t* srcSeqH,
    std::size_t* tgtSeqH) noexcept :
    
    norm1(input, out1, gradient1, inputGradient),
    mulAtt1(out1, out1, out1, out2, gradient2, gradient1, gradient1, gradient1, MaskType::LOOK_AHEAD, tgtSeqH),
    dropout1(out2, out3, gradient3, gradient2, batch * sequenceLength, dModel),
    norm2(out3, out4, gradient4, gradient3),
    mulAtt2(out4, encoderOut, encoderOut, out5, gradient5, gradient4, encoderGradient, encoderGradient, MaskType::CROSS_PADDING, srcSeqH),
    dropout2(out5, out6, gradient6, gradient5, batch * sequenceLength, dModel),
    norm3(out6, out7, gradient7, gradient6),
    pff(out7, out8, gradient8, gradient7),
    dropout3(out8, output, outputGradient, gradient8, batch * sequenceLength, dModel),

    input(input),
    encoderOut(encoderOut),
    output(output),
    outputGradient(outputGradient),
    inputGradient(inputGradient),
    encoderGradient(encoderGradient),
    srcSeqH(srcSeqH),
    tgtSeqH(tgtSeqH),

    out1(batch * sequenceLength, dModel),
    out2(batch * sequenceLength, dModel),
    out3(batch * sequenceLength, dModel),
    out4(batch * sequenceLength, dModel),
    out5(batch * sequenceLength, dModel),
    out6(batch * sequenceLength, dModel),
    out7(batch * sequenceLength, dModel),
    out8(batch * sequenceLength, dModel),

    gradient1(batch * sequenceLength, dModel),
    gradient2(batch * sequenceLength, dModel),
    gradient3(batch * sequenceLength, dModel),
    gradient4(batch * sequenceLength, dModel),
    gradient5(batch * sequenceLength, dModel),
    gradient6(batch * sequenceLength, dModel),
    gradient7(batch * sequenceLength, dModel),
    gradient8(batch * sequenceLength, dModel) {;}

void DecoderLayer::UpdateGraph(cudaGraphExec_t graphExec) {
    mulAtt1.UpdateGraph();
    mulAtt2.UpdateGraph();
}

cudaGraphNode_t DecoderLayer::AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = norm1.AppendGraphForward(graph, dependencyNodes);
    cudaGraphNode_t k2 = mulAtt1.AppendGraphForward(graph, { k1 });
    cudaGraphNode_t k3 = dropout1.AppendGraphForward(graph, { k2 });
    cudaGraphNode_t k4 = AppendPlusInplaceNode(graph, { k3 }, out3, input);
    cudaGraphNode_t k5 = norm2.AppendGraphForward(graph, { k4 });
    cudaGraphNode_t k6 = mulAtt2.AppendGraphForward(graph, { k5 });
    cudaGraphNode_t k7 = dropout2.AppendGraphForward(graph, { k6 });
    cudaGraphNode_t k8 = AppendPlusInplaceNode(graph, { k7 }, out6, out3);
    cudaGraphNode_t k9 = norm3.AppendGraphForward(graph, { k8 });
    cudaGraphNode_t k10 = pff.AppendGraphForward(graph, { k9 });
    cudaGraphNode_t k11 = dropout3.AppendGraphForward(graph, { k10 });
    cudaGraphNode_t k12 = AppendPlusInplaceNode(graph, { k11 }, output, out6);
    return k12;
}

cudaGraphNode_t DecoderLayer::AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = norm1.AppendGraphPredict(graph, dependencyNodes);
    cudaGraphNode_t k2 = mulAtt1.AppendGraphPredict(graph, { k1 });
    cudaGraphNode_t k3 = dropout1.AppendGraphPredict(graph, { k2 });
    cudaGraphNode_t k4 = AppendPlusInplaceNode(graph, { k3 }, out3, input);
    cudaGraphNode_t k5 = norm2.AppendGraphPredict(graph, { k4 });
    cudaGraphNode_t k6 = mulAtt2.AppendGraphPredict(graph, { k5 });
    cudaGraphNode_t k7 = dropout2.AppendGraphPredict(graph, { k6 });
    cudaGraphNode_t k8 = AppendPlusInplaceNode(graph, { k7 }, out6, out3);
    cudaGraphNode_t k9 = norm3.AppendGraphPredict(graph, { k8 });
    cudaGraphNode_t k10 = pff.AppendGraphPredict(graph, { k9 });
    cudaGraphNode_t k11 = dropout3.AppendGraphPredict(graph, { k10 });
    cudaGraphNode_t k12 = AppendPlusInplaceNode(graph, { k11 }, output, out6);
    return k12;
}

cudaGraphNode_t DecoderLayer::AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = dropout3.AppendGraphBackward(graph, dependencyNodes);
    cudaGraphNode_t k2 = pff.AppendGraphBackward(graph, { k1 });
    cudaGraphNode_t k3 = norm3.AppendGraphBackward(graph, { k2 });
    cudaGraphNode_t k4 = AppendPlusInplaceNode(graph, { k3 }, gradient6, outputGradient);
    cudaGraphNode_t k5 = dropout2.AppendGraphBackward(graph, { k4 });
    cudaGraphNode_t k6 = mulAtt2.AppendGraphBackward(graph, { k5 });
    cudaGraphNode_t k7 = norm2.AppendGraphBackward(graph, { k6 });
    cudaGraphNode_t k8 = AppendPlusInplaceNode(graph, { k7 }, gradient3, gradient6);
    cudaGraphNode_t k9 = dropout1.AppendGraphBackward(graph, { k8 });
    cudaGraphNode_t k10 = mulAtt1.AppendGraphBackward(graph, { k9 });
    cudaGraphNode_t k11 = norm1.AppendGraphBackward(graph, { k10 });
    cudaGraphNode_t k12 = AppendPlusInplaceNode(graph, { k11 }, inputGradient, gradient3);
    return k12;
}

cudaGraphNode_t DecoderLayer::AppendGraphUpdateParameter(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = SyncDependency(graph, dependencyNodes);
    cudaGraphNode_t k2 = norm1.AppendGraphUpdateParameter(graph, {k1});
    cudaGraphNode_t k3 = mulAtt1.AppendGraphUpdateParameter(graph, {k1});
    cudaGraphNode_t k4 = norm2.AppendGraphUpdateParameter(graph, {k1});
    cudaGraphNode_t k5 = mulAtt2.AppendGraphUpdateParameter(graph, {k1});
    cudaGraphNode_t k6 = norm3.AppendGraphUpdateParameter(graph, {k1});
    cudaGraphNode_t k7 = pff.AppendGraphUpdateParameter(graph, {k1});
    cudaGraphNode_t k8 = SyncDependency(graph, {k2,k3,k4,k5,k6,k7});
    return k8;
}

void DecoderLayer::loadParam(cnpy::npz_t npFile, std::string prefix) {
    norm1.loadParam(npFile, prefix + ".sub1.layerNorm");
    mulAtt1.loadParam(npFile, prefix + ".sub1.sublayer");
    norm2.loadParam(npFile, prefix + ".sub2.layerNorm");
    mulAtt2.loadParam(npFile, prefix + ".sub2.sublayer");
    norm3.loadParam(npFile, prefix + ".sub3.layerNorm");
    pff.loadParam(npFile,prefix + ".sub3.sublayer");
}

void DecoderLayer::checkUpdatedParam(cnpy::npz_t npFile, std::string prefix) {
    norm1.checkUpdatedParam(npFile, prefix + ".sub1.layerNorm");
    mulAtt1.checkUpdatedParam(npFile, prefix + ".sub1.sublayer");
    norm2.checkUpdatedParam(npFile, prefix + ".sub2.layerNorm");
    mulAtt2.checkUpdatedParam(npFile, prefix + ".sub2.sublayer");
    norm3.checkUpdatedParam(npFile, prefix + ".sub3.layerNorm");
    pff.checkUpdatedParam(npFile, prefix + ".sub3.sublayer");
}

void DecoderLayer::forwardTest(cnpy::npz_t npFile, std::string prefix) {
    Tensor target(output.row, output.col);

    target.loadNp(npFile, prefix + ".output");
    input.loadNp(npFile, prefix + ".input1");
    encoderOut.loadNp(npFile, prefix + ".input2");
	Tensor npdLoader(1, 2);
	npdLoader.loadNp(npFile, prefix + ".npd");
	float* _SeqH = new float[batch];
	npdLoader.toFloat((float*)_SeqH);
	for(int i = 0;i < batch;i++) {
		srcSeqH[i] = _SeqH[0];
	}
    for(int i = 0;i < batch;i++) {
		tgtSeqH[i] = _SeqH[1];
	}

    // Forward
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaGraphCreate(&graph, 0);
    this->AppendGraphForward(graph, {});
    cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
	this->UpdateGraph(instance);
    cudaGraphLaunch(instance, 0);
    cudaDeviceSynchronize();

    PrintTestResult("forward", output, target);
}

void DecoderLayer::backwardTest(cnpy::npz_t npFile, std::string prefix) {
    Set(outputGradient, 1.0f / output.row / output.col);
    cudaDeviceSynchronize();

    // load input
    input.loadNp(npFile, prefix + ".input1");
    encoderOut.loadNp(npFile, prefix + ".input2");
	Tensor npdLoader(1, 2);
	npdLoader.loadNp(npFile, prefix + ".npd");
	float* _SeqH = new float[batch];
	npdLoader.toFloat((float*)_SeqH);
	for(int i = 0;i < batch;i++) {
		srcSeqH[i] = _SeqH[0];
	}
    for(int i = 0;i < batch;i++) {
		tgtSeqH[i] = _SeqH[1];
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
    this->UpdateGraph(instance);
    cudaGraphLaunch(instance, 0);
    cudaDeviceSynchronize();
    // Print(gradient1, 0, 0, 10, 10);
    // Print(gradient2, 0, 0, 10, 10);
    // Print(gradient3, 0, 0, 10, 10);
    // Print(linear1.biasOpt.gradient, 0, 0, 10 ,10);
    
    checkUpdatedParam(npFile, prefix);
}