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
    Tensor input,
    Tensor output,
    Tensor outputGradient,
    Tensor inputGradient) noexcept :

    input()


void Encoder::UpdateGraph(cudaGraphExec_t graphExec);

cudaGraphNode_t Encoder::AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

cudaGraphNode_t Encoder::AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

cudaGraphNode_t Encoder::AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

cudaGraphNode_t Encoder::AppendGraphUpdateParameter(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

void Encoder::loadParam(cnpy::npz_t npFile, std::string prefix);

void Encoder::checkUpdatedParam(cnpy::npz_t npFile, std::string prefix);

void Encoder::forwardTest(cnpy::npz_t npFile, std::string prefix);

void Encoder::backwardTest(cnpy::npz_t npFile, std::string prefix);
