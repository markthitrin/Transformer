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
    ResetGraph();
    delete[] inputEncoderH;
    delete[] inputDecoderH;
    delete[] srcSeqH;
    delete[] tgtSeqH;
}

float Transformer::Train(const std::size_t* encoderInput, const std::size_t* srcSeq,
    const std::size_t* decoderInput, const std::size_t* tgtSeq,
    const std::size_t* targetOutput) {

    for(int i = 0;i < batch * sequenceLength;i++) {
        this->inputEncoderH[i] = encoderInput[i];
        this->inputDecoderH[i] = decoderInput[i];
    }
    for(int i = 0;i < batch;i++) {
        this->srcSeqH[i] = srcSeq[i];
        this->tgtSeqH[i] = tgtSeq[i];
    }
    
    SetTrainGraph();
    UpdateGraph(graphExecTrain);
    cudaGraphLaunch(graphExecTrain, 0);
    cudaStreamSynchronize(0);

    float loss = CrossEntropy(output, targetOutput, outputGradient, tgtSeqH);

    return loss;
}

void Transformer::Encode(const std::size_t* encoderInput, const std::size_t* srcSeq) {
    for(int i = 0;i < batch * sequenceLength;i++) {
        this->inputEncoderH[i] = encoderInput[i];
    }
    for(int i = 0;i < batch;i++) {
        this->srcSeqH[i] = srcSeq[i];
    }

    SetPredictGraph();
    UpdateGraph(graphExecEncode);
    cudaGraphLaunch(graphExecEncode, 0);
    cudaStreamSynchronize(0);
}

void Transformer::Decode(const std::size_t* decoderInput, const std::size_t* tgtSeq) {
    for(int i = 0;i < batch * sequenceLength;i++) {
        this->inputDecoderH[i] = decoderInput[i];
    }
    for(int i = 0;i < batch;i++) {
        this->tgtSeqH[i] = tgtSeq[i];
    }

    SetPredictGraph();
    UpdateGraph(graphExecDecode);
    cudaGraphLaunch(graphExecDecode, 0);
    cudaStreamSynchronize(0);
}

void Transformer::ResetGraph() {
    if(graphState == 1) {
        cudaGraphDestroy(graphTrain);
        cudaGraphExecDestroy(graphExecTrain);
    }
    if(graphState == 2) {
        cudaGraphDestroy(graphEncode);
        cudaGraphExecDestroy(graphExecEncode);
        cudaGraphDestroy(graphDecode);
        cudaGraphExecDestroy(graphExecDecode);
    }
}

void Transformer::SetTrainGraph() {
    if(graphState != 1) {
        std::cout << "Transformer : Changing to training graph...\n";
        ResetGraph();
        std::cout << "Transformer : Building graph...\n";
        cudaError_t err = cudaGraphCreate(&graphTrain, 0);
        cudaGraphNode_t k1 = AppendGraphForward(graphTrain, {});
        cudaGraphNode_t k2 = AppendGraphBackward(graphTrain, {k1});
        cudaGraphNode_t k3 = AppendGraphUpdateParameter(graphTrain, {k2});
        std::cout << "Transformer : Instantiating graphexec...\n";
        cudaGraphInstantiate(&graphExecTrain, graphTrain, nullptr, nullptr, 0);
        graphState = 1;
    } 
}

void Transformer::SetPredictGraph() {
    if(graphState != 2) {
        std::cout << "Transformer : Changing to evaluation graph...\n";
        ResetGraph();

        std::cout << "Transformer : Building encode graph...\n";
        cudaGraphCreate(&graphEncode, 0);
        cudaGraphNode_t k1 = AppendGraphPredictEncode(graphTrain, {});
        std::cout << "Transformer : Instantiating graphexec...\n";
        cudaGraphInstantiate(&graphExecEncode, graphEncode, nullptr, nullptr, 0);

        std::cout << "Transformer : Building decode graph...\n";
        cudaGraphCreate(&graphDecode, 0);
        cudaGraphNode_t k2 = AppendGraphPredictDecode(graphDecode, {});
        std::cout << "Transformer : Instantiating graphexec...\n";
        cudaGraphInstantiate(&graphExecDecode, graphDecode, nullptr, nullptr, 0);
        graphState = 2;
    }
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

cudaGraphNode_t Transformer::AppendGraphPredictEncode(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = srcEmbed.AppendGraphPredict(graph, dependencyNodes);
    cudaGraphNode_t k2 = srcPos.AppendGraphPredict(graph, { k1 });
    cudaGraphNode_t k3 = encoder.AppendGraphPredict(graph, { k2 });
    return k3;
}

cudaGraphNode_t Transformer::AppendGraphPredictDecode(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = tgtEmbed.AppendGraphPredict(graph, dependencyNodes);
    cudaGraphNode_t k2 = tgtPos.AppendGraphPredict(graph, { k1 });
    cudaGraphNode_t k3 = decoder.AppendGraphPredict(graph, { k2 });
    cudaGraphNode_t k4 = linear.AppendGraphForward(graph, { k3 });
    return k4;
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