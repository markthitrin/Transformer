#ifndef UTIL
#define UTIL

#include "Header.cuh"
#include "Tensor.cuh"
#include "UtilKernel.cuh"

class AdamOptimizer {
public:
    AdamOptimizer(const Tensor& param);
    AdamOptimizer(const std::size_t row, const std::size_t col);
    AdamOptimizer(const AdamOptimizer& other);
    AdamOptimizer(AdamOptimizer&& other);
    ~AdamOptimizer();

    Tensor gradient;
    Tensor accM;
    Tensor accV;
    std::size_t* t;
};



void AdamOpt(Tensor& param, AdamOptimizer opt);
cudaGraphNode_t AppendAdamOptNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& param, AdamOptimizer& opt);

float CrossEntropy(Tensor& logits, const std::size_t* targetH, Tensor& gradient, std::size_t* tgtSeqH);

void Print(Tensor& A, const std::size_t r0, const std::size_t c0, const std::size_t r, const std::size_t c);

void CrossEntropyF(Tensor& logits, Tensor& sumExp, Tensor& maxValue, Tensor& gradient, const std::size_t* target, const bool* tgtSeqHot, Tensor& loss);

void SoftmaxF(Tensor& input, Tensor& sumExp, Tensor& maxValue, Tensor& output, const bool* tgtSeqHot);

cudaGraphNode_t SyncDependency(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes = {});

#endif