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

void Decoder::loadParam(cnpy::npz_t npFile, std::string prefix) {
    for(int i = 0;i < N;i++) {
        layers[i]->loadParam(npFile, prefix + ".layer" + std::to_string(i));
    }
    norm->loadParam(npFile, prefix + ".norm");
}

void Decoder::checkUpdatedParam(cnpy::npz_t npFile, std::string prefix) {
    for(int i = 0;i < N;i++) {
        layers[i]->checkUpdatedParam(npFile, prefix + ".layer" + std::to_string(i));
    }
    norm->checkUpdatedParam(npFile, prefix + ".norm");
}

void Decoder::forwardTest(cnpy::npz_t npFile, std::string prefix) {
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
	this->UpdateGraph();
    cudaGraphLaunch(instance, 0);
    cudaDeviceSynchronize();

    PrintTestResult("forward", output, target);
}

void Decoder::backwardTest(cnpy::npz_t npFile, std::string prefix) {
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
    this->UpdateGraph();
    cudaGraphLaunch(instance, 0);
    cudaDeviceSynchronize();

    // Print(gradient1, 0, 0, 10, 10);
    // Print(gradient2, 0, 0, 10, 10);
    // Print(gradient3, 0, 0, 10, 10);
    // Print(linear1.biasOpt.gradient, 0, 0, 10 ,10);
    
    checkUpdatedParam(npFile, prefix);
}
