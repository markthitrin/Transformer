#ifndef UTIL
#define UTIL

#include "Header.cuh"
#include "Tensor.cuh"

class AdamOptimizer {
public:
    AdamOptimizer(Tensor param);
    AdamOptimizer(const std::size_t row, const std::size_t col);
    AdamOptimizer(AdamOptimizer& other);
    ~AdamOptimizer();

    Tensor gradient;
    Tensor accM;
    Tensor accV;
    int t;
};

__global__ void AdamOptKernel(
    const float* param, const float* gradient, const float* accM, const float* accV, const float t,
    const std::size_t pitchParam, const std::size_t pitchGrad, const std::size_t pitchAccM, const std::size_t pitchAccV,
    const std::size_t row, const std::size_t col);
void AdamOpt(Tensor param, AdamOptimizer opt, const int feedCount = 1);
cudaGraphNode_t AppendAdamOptNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes = {},
    Tensor param, AdamOptimizer opt, int feedCount = 1);

float CrossEntropy(Tensor logits, Tensor target, Tensor gradient, int npd[batch]);
float fast_logf(float x);

void PrintTestResult(std::string text, Tensor A, Tensor B);

void PrintTestResultT(std::string text, Tensor A, Tensor B);

cudaGraphNode_t SyncDependency(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes = {});

#endif