#ifndef TENSOR_GRAPH
#define TENSOR_GRAPH

#include "Header.cuh"
#include "Tensor.cuh"

cudaGraphNode_t AppendCopyBatchNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor A, Tensor B, const std::size_t batch);

cudaGraphNode_t AppendPlusBatchNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor A, Tensor B, Tensor C,
    std::size_t batch);
cudaGraphNode_t AppendPlusInplceBatchNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor A, Tensor C,
    std::size_t batch);

cudaGraphNode_t AppendMulNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor A, const float x, Tensor C);
cudaGraphNode_t AppendMulInplaceNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor A, const float x);

cudaGraphNode_t AppendResetNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor A);

cudaGraphNode_t AppendMatMulPlusNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor A, Tensor B, Tensor C, bool ATransposed, bool BTransposed);

#endif