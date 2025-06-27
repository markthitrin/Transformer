#ifndef TENSOR_GRAPH
#define TENSOR_GRAPH

#include "Header.cuh"
#include "Tensor.cuh"

cudaGraphNode_t AppendCopyNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor A, Tensor C);
cudaGraphNode_t AppendCopyBatchNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor A, Tensor B, const std::size_t batch);

cudaGraphNode_t AppendPlusBatchNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor A, Tensor B, Tensor C,
    std::size_t batch);
cudaGraphNode_t AppendPlusInplaceBatchNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor A, Tensor B,
    std::size_t batch);
cudaGraphNode_t AppendPlusReduceInplceBatchNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor A, Tensor C,
    std::size_t batch);
cudaGraphNode_t AppendPlusProductReduceInplceBatchNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor A, Tensor B, Tensor C,
    std::size_t batch);

cudaGraphNode_t AppendMulNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor A, const float x, Tensor C);
cudaGraphNode_t AppendMulInplaceNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor A, const float x);

cudaGraphNode_t AppendDivNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor A, const float x, Tensor C);
cudaGraphNode_t AppendDivInplaceNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor A, const float x);

cudaGraphNode_t AppendResetNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor A);

cudaGraphNode_t AppendReduceSumOfProductNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor A, Tensor B, Tensor sumOfProduct);
cudaGraphNode_t AppendReduceSumNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor input, Tensor sum);
cudaGraphNode_t AppendReduceMaxNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor input, Tensor maxValue);

cudaGraphNode_t AppendMeanNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor input, Tensor mean);
cudaGraphNode_t AppendStdNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor input, Tensor mean, Tensor std);

cudaGraphNode_t AppendLookAheadMaskBatchNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor Attention, const std::size_t batch, std::size_t* seq, const float x);
cudaGraphNode_t AppendPaddingMaskBatchNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor Attention, const std::size_t batch, std::size_t* seq, const float x);
cudaGraphNode_t AppendCrossPaddingMaskBatchNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor Attention, const std::size_t batch, std::size_t* seq, const float x);

cudaGraphNode_t AppendMatMulPlusNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor A, Tensor B, Tensor C, bool ATransposed, bool BTransposed);
cudaGraphNode_t AppendMatMulPlusBatchNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor A, Tensor B, Tensor C, const bool ATransposed, const bool BTransposed,
    const std::size_t batch, const bool ABroadcast, const bool BBroadcast, const bool CBroadcast);

#endif