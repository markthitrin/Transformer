#include "Header.cuh"
#include "Tensor.cuh"
#include "EncoderLayer.cuh"
#include "Softmax.cuh"
#include "PositionalEncoder.cuh"
#include "Embedding.cuh"
#include "Linear.cuh"
#include "LayerNorm.cuh"
#include "Util.cuh"
#include "Encoder.cuh"


Encoder::Encoder(
    Tensor& input,
    Tensor& output,
    Tensor& outputGradient,
    Tensor& inputGradient,
    std::size_t* srcSeqH) noexcept :
    
    input(input),
    output(output),
    outputGradient(outputGradient),
    inputGradient(inputGradient),
    srcSeqH(srcSeqH) {

    out.reserve(N);
    gradient.reserve(N);
    for(int i = 0;i < N;i++) {
        out.emplace_back(batch * sequenceLength, dModel);
        gradient.emplace_back(batch * sequenceLength, dModel);
    }
    
    norm = new LayerNorm(out[N - 1], output, outputGradient, gradient[N - 1]);
    layers[0] = new EncoderLayer(input, out[0], gradient[0], inputGradient, srcSeqH);
    for(int i = 1;i < N;i++) {
        layers[i] = new EncoderLayer(out[i - 1], out[i], gradient[i], gradient[i - 1], srcSeqH);
    }
}
Encoder::~Encoder() {
    delete norm;
    for(int i = 0 ;i < N;i++) {
        delete layers[i];
    }
}

void Encoder::UpdateGraph() {
    for(int i = 0;i < N;i++) {
        layers[i]->UpdateGraph();
    }
}

cudaGraphNode_t Encoder::AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1[N];
    k1[0] = layers[0]->AppendGraphForward(graph, dependencyNodes);
    for(int i = 1;i < N;i++) {
        k1[i] = layers[i]->AppendGraphForward(graph, { k1[i - 1] });
    }
    cudaGraphNode_t k2 = norm->AppendGraphForward(graph, { k1[N - 1] });
    return k2;
}

cudaGraphNode_t Encoder::AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1[N];
    k1[0] = layers[0]->AppendGraphPredict(graph, dependencyNodes);
    for(int i = 1;i < N;i++) {
        k1[i] = layers[i]->AppendGraphPredict(graph, { k1[i - 1] });
    }
    cudaGraphNode_t k2 = norm->AppendGraphPredict(graph, { k1[N - 1] });
    return k2;
}

cudaGraphNode_t Encoder::AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = norm->AppendGraphBackward(graph, dependencyNodes);
    cudaGraphNode_t k2[N];
    k2[N - 1] = layers[N - 1]->AppendGraphBackward(graph, { k1 });
    for(int i = N - 2;i >= 0;i--) {
        k2[i] = layers[i]->AppendGraphBackward(graph, { k2[i + 1] });
    }
    return k2[0];
}

cudaGraphNode_t Encoder::AppendGraphUpdateParameter(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
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

void Encoder::loadParam(cnpy::npz_t npFile, std::string prefix) {
    for(int i = 0;i < N;i++) {
        layers[i]->loadParam(npFile, prefix + ".layer" + std::to_string(i));
    }
    norm->loadParam(npFile, prefix + ".norm");
}

void Encoder::checkUpdatedParam(cnpy::npz_t npFile, std::string prefix) {
    for(int i = 0;i < N;i++) {
        layers[i]->checkUpdatedParam(npFile, prefix + ".layer" + std::to_string(i));
    }
    norm->checkUpdatedParam(npFile, prefix + ".norm");
}

void Encoder::forwardTest(cnpy::npz_t npFile, std::string prefix) {
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

void Encoder::backwardTest(cnpy::npz_t npFile, std::string prefix) {
    Tensor target(output.row, output.col);
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
    target.loadNp(npFile, prefix + ".output");

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
