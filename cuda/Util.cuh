#ifndef UTIL
#define UTIL

#include "Header.cuh"
#include "Tensor.cuh"
#include "UtilKernel.cuh"

class AdamOptimizer {
public:
    AdamOptimizer(Tensor param);
    AdamOptimizer(const std::size_t row, const std::size_t col);
    AdamOptimizer(const AdamOptimizer& other);
    ~AdamOptimizer();

    Tensor gradient;
    Tensor accM;
    Tensor accV;
    std::size_t* t;
};

void AdamOpt(Tensor param, AdamOptimizer opt);
cudaGraphNode_t AppendAdamOptNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor param, AdamOptimizer opt);

float CrossEntropy(Tensor logits, Tensor target, Tensor gradient, int npd[batch]);
float fast_logf(float x);

void Print(Tensor A, const std::size_t r0, const std::size_t c0, const std::size_t r, const std::size_t c);

void PrintTestResult(std::string text, Tensor A, Tensor B);

void PrintTestResultT(std::string text, Tensor A, Tensor B);

cudaGraphNode_t SyncDependency(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes = {});

#endif