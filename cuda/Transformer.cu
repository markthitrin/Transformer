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
#include "Transformer.cuh"

Transformer::Transformer() noexcept :
    srcEmbed(inputEncoderH, out1, gradient1, srcVocab),
    tgtEmbed(inputDecoderH, out2, gradient2, tgtVocab),
    srcPos(out1, out3, gradient3, gradient1),
    tgtPos(out2, out4, gradient4, gradient2),
    encoder(out3, encoderOut, encoderGradient, gradient3, srcSeqH),
    decoder(out4, encoderOut, out5, gradient5, gradient4, encoderGradient, srcSeqH, tgtSeqH),
    linear(out5, output, outputGradient, gradient5, dModel, tgtVocab),

    inputEncoderH(new std::size_t[batch * sequenceLength]),
    inputDecoderH(new std::size_t[batch * sequenceLength]),
    srcSeqH(new std::size_t[batch]),
    tgtSeqH(new std::size_t[batch]),
    output(batch * sequenceLength, tgtVocab),
    outputGradient(batch * sequenceLength, tgtVocab),

    encoderOut(batch * sequenceLength, dModel),
    encoderGradient(batch * sequenceLength, dModel),
    
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

Transformer::~Transformer() {
    delete[] inputEncoderH;
    delete[] inputDecoderH;
    delete[] srcSeqH;
    delete[] tgtSeqH;
}

void Transformer::UpdateGraph(cudaGraphExec_t instance) {
	srcEmbed.UpdateGraph(instance);
    tgtEmbed.UpdateGraph(instance);
    encoder.UpdateGraph();
    decoder.UpdateGraph();
}

cudaGraphNode_t Transformer::AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = srcEmbed.AppendGraphForward(graph, dependencyNodes);
    cudaGraphNode_t k2 = tgtEmbed.AppendGraphForward(graph, dependencyNodes);
    cudaGraphNode_t k3 = srcPos.AppendGraphForward(graph, { k1 });
    cudaGraphNode_t k4 = tgtPos.AppendGraphForward(graph, { k2 });
    cudaGraphNode_t k5 = encoder.AppendGraphForward(graph, { k3 });
    cudaGraphNode_t k6 = decoder.AppendGraphForward(graph, { k4, k5 });
    cudaGraphNode_t k7 = linear.AppendGraphForward(graph, { k6 });
    return k7;
}

cudaGraphNode_t Transformer::AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = srcEmbed.AppendGraphPredict(graph, dependencyNodes);
    cudaGraphNode_t k2 = tgtEmbed.AppendGraphPredict(graph, dependencyNodes);
    cudaGraphNode_t k3 = srcPos.AppendGraphPredict(graph, { k1 });
    cudaGraphNode_t k4 = tgtPos.AppendGraphPredict(graph, { k2 });
    cudaGraphNode_t k5 = encoder.AppendGraphPredict(graph, { k3 });
    cudaGraphNode_t k6 = decoder.AppendGraphPredict(graph, { k4, k5 });
    cudaGraphNode_t k7 = linear.AppendGraphPredict(graph, { k6 });
    return k7;
}

cudaGraphNode_t Transformer::AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = linear.AppendGraphBackward(graph, dependencyNodes);
    cudaGraphNode_t k2 = decoder.AppendGraphBackward(graph, { k1 });
    cudaGraphNode_t k3 = encoder.AppendGraphBackward(graph, { k2 });
    cudaGraphNode_t k4 = tgtPos.AppendGraphBackward(graph, { k2 });
    cudaGraphNode_t k5 = srcPos.AppendGraphBackward(graph, { k3 });
    cudaGraphNode_t k6 = tgtEmbed.AppendGraphBackward(graph, { k4 });
    cudaGraphNode_t k7 = srcEmbed.AppendGraphBackward(graph, { k5 });
    cudaGraphNode_t k8 = SyncDependency(graph, { k6, k7 });
    return k8;
}

cudaGraphNode_t Transformer::AppendGraphUpdateParameter(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = SyncDependency(graph, dependencyNodes);
    cudaGraphNode_t k2 = linear.AppendGraphUpdateParameter(graph, { k1 });
    cudaGraphNode_t k3 = decoder.AppendGraphUpdateParameter(graph, { k1 });
    cudaGraphNode_t k4 = encoder.AppendGraphUpdateParameter(graph, { k1 });
    cudaGraphNode_t k5 = tgtEmbed.AppendGraphUpdateParameter(graph, { k1 });
    cudaGraphNode_t k6 = srcEmbed.AppendGraphUpdateParameter(graph, { k1 });
    cudaGraphNode_t k7 = SyncDependency(graph, { k2, k3, k4, k5, k6 });
    return k7;
}

void Transformer::loadParam(cnpy::npz_t npFile, std::string prefix) {
    encoder.loadParam(npFile, prefix + ".encoder");
    decoder.loadParam(npFile, prefix + ".decoder");
    srcEmbed.loadParam(npFile, prefix + ".src_embed");
    tgtEmbed.loadParam(npFile, prefix + ".tgt_embed");
    linear.weight.loadNp(npFile, prefix + ".projection_layer.weight");
    linear.bias.loadNp(npFile, prefix + ".projection_layer.bias");
}

void Transformer::checkUpdatedParam(cnpy::npz_t npFile, std::string prefix) {
    encoder.checkUpdatedParam(npFile, prefix + ".encoder");
    decoder.checkUpdatedParam(npFile, prefix + ".decoder");
    srcEmbed.checkUpdatedParam(npFile, prefix + ".src_embed");
    tgtEmbed.checkUpdatedParam(npFile, prefix + ".tgt_embed");
    linear.checkUpdatedParam(npFile, prefix + ".projection_layer");
}

void Transformer::forwardTest(cnpy::npz_t npFile, std::string prefix) {
    Tensor target(batch * sequenceLength, tgtVocab);
    Tensor targetd(batch * sequenceLength, dModel);
    Tensor ttt(batch * sequenceLength, dModel);
    Tensor _inputEncoderH(1, batch * sequenceLength);
    Tensor _inputDecoderH(1, batch * sequenceLength);
    Tensor npdLoader(1, 2);

    _inputEncoderH.loadNp(npFile, prefix + ".input1");
    _inputDecoderH.loadNp(npFile, prefix + ".input2");
    target.loadNp(npFile, prefix + ".output");
    targetd.loadNp(npFile, prefix + ".outputd");
    ttt.loadNp(npFile, prefix + ".layer0.sub1.output");
    npdLoader.loadNp(npFile, prefix + ".npd");

    float* encoderH = new float[batch * sequenceLength];
    float* decoderH = new float[batch * sequenceLength];
    float* _seqH = new float[batch];
    _inputEncoderH.toFloat(encoderH);
    _inputDecoderH.toFloat(decoderH);
	npdLoader.toFloat((float*)_seqH);

	for(int i = 0;i < batch;i++) {
		srcSeqH[i] = _seqH[0];
	}
    for(int i = 0;i < batch;i++) {
		tgtSeqH[i] = _seqH[1];
	}
    for(int i = 0;i < batch * sequenceLength;i++) {
        inputEncoderH[i] = encoderH[i];
        inputDecoderH[i] = decoderH[i];
    }

    // Forward
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaGraphCreate(&graph, 0);
    this->AppendGraphForward(graph, {});
    cudaError_t err = cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
	this->UpdateGraph(instance);
    cudaGraphLaunch(instance, 0);
    cudaDeviceSynchronize();

    PrintTestResult("forward", output, target);

    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(instance);
}

void Transformer::backwardTest(cnpy::npz_t npFile, std::string prefix) {
    Set(outputGradient, 1.0f / batch / sequenceLength / tgtVocab);
    Tensor target(batch * sequenceLength, tgtVocab);
    Tensor targete(batch * sequenceLength, dModel);
    Tensor _inputEncoderH(1, batch * sequenceLength);
    Tensor _inputDecoderH(1, batch * sequenceLength);
    Tensor npdLoader(1, 2);

    _inputEncoderH.loadNp(npFile, prefix + ".input1");
    _inputDecoderH.loadNp(npFile, prefix + ".input2");
    target.loadNp(npFile, prefix + ".output");
    targete.loadNp(npFile, prefix + ".outpute");
    npdLoader.loadNp(npFile, prefix + ".npd");

    float* encoderH = new float[batch * sequenceLength];
    float* decoderH = new float[batch * sequenceLength];
    float* _seqH = new float[batch];
    _inputEncoderH.toFloat(encoderH);
    _inputDecoderH.toFloat(decoderH);
	npdLoader.toFloat((float*)_seqH);

	for(int i = 0;i < batch;i++) {
		srcSeqH[i] = _seqH[0];
	}
    for(int i = 0;i < batch;i++) {
		tgtSeqH[i] = _seqH[1];
	}
    for(int i = 0;i < batch * sequenceLength;i++) {
        inputEncoderH[i] = encoderH[i];
        inputDecoderH[i] = decoderH[i];
    }

    // Forward, backward, update
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaGraphCreate(&graph, 0);
    cudaGraphNode_t k1 = this->AppendGraphForward(graph, {});
    cudaGraphNode_t k2 = this->AppendGraphBackward(graph, {k1});
    this->AppendGraphUpdateParameter(graph, {k2});
    // cudaGraphDebugDotPrint(graph, "graph.dot", cudaGraphDebugDotFlagsVerbose);
    cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
	this->UpdateGraph(instance);
    cudaGraphLaunch(instance, 0);
    cudaDeviceSynchronize();

    checkUpdatedParam(npFile, prefix);

    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(instance);
}