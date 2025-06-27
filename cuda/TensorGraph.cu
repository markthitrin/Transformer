#include "Header.cuh"
#include "Tensor.cuh"
#include "TensorKernel.cuh"
#include "Util.cuh"
#include "UtilKernel.cuh"

cudaGraphNode_t AppendCopyNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& A, Tensor& C) {
    
    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    cudaMemcpy3DParms copyParams = {};
    copyParams.srcPtr = make_cudaPitchedPtr(A.data, A.pitch, A.col, A.row);
    copyParams.dstPtr = make_cudaPitchedPtr(C.data, C.pitch, C.col, C.row);
    copyParams.extent = make_cudaExtent(sizeof(float) * A.col, A.row, 1);
    copyParams.kind = cudaMemcpyDeviceToDevice;

    cudaGraphNode_t copyNode;
    cudaGraphAddMemcpyNode(&copyNode, graph, &dependency, numDependency, &copyParams);

    return copyNode;
}
cudaGraphNode_t AppendCopyBatchNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& A, Tensor& C, const std::size_t batch) {
    
    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    const int sr = C.row / batch;
    std::vector<cudaGraphNode_t> nodes(batch);
    for(int i = 0;i < batch;i++) {
        cudaMemcpy3DParms copyParams = {};
        copyParams.srcPtr = make_cudaPitchedPtr(A.data, A.pitch, A.col, A.row);
        copyParams.dstPtr = make_cudaPitchedPtr(GetRow(C.data, i * sr, C.pitch), C.pitch, A.col, A.row);
        copyParams.extent = make_cudaExtent(sizeof(float) * A.col, A.row, 1);
        copyParams.kind = cudaMemcpyDeviceToDevice;
        cudaGraphAddMemcpyNode(&nodes[i], graph, &dependency, numDependency, &copyParams);
    }

    return SyncDependency(graph, nodes);
}

cudaGraphNode_t AppendPlusNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& A, Tensor& B, Tensor& C) {
    
    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(C.col, BLOCKSIZE), ceil(C.row, BLOCKSIZE));

    void* kernelArgs[] = { &A.data, &B.data, &C.data, &A.pitch, &B.pitch, &C.pitch, &C.row, &C.col};

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)static_cast<void(*)(
                                const float*, const float*, float*,
                                const std::size_t, const std::size_t, const std::size_t,
                                const std::size_t, const std::size_t)>(PlusKernel);
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = nullptr;

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);

    return kernelNode;
}
cudaGraphNode_t AppendPlusInplaceNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& A, Tensor& B) {

    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(A.col, BLOCKSIZE), ceil(A.row, BLOCKSIZE));

    void* kernelArgs[] = { &A.data, &B.data, &A.pitch, &B.pitch, &A.row, &A.col};

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)static_cast<void(*)(
                                float*, const float*,
                                const std::size_t, const std::size_t,
                                const std::size_t, const std::size_t)>(PlusInplaceKernel);
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = nullptr;

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);

    return kernelNode;
}
cudaGraphNode_t AppendPlusBatchNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& A, Tensor& B, Tensor& C,
    std::size_t batch) {
    
    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(B.col, BLOCKSIZE), ceil(B.row, BLOCKSIZE));

    void* kernelArgs[] = { &A.data, &B.data, &C.data, &A.pitch, &B.pitch, &C.pitch, &batch, &B.row, &C.col};

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)PlusBatchKernel;
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = nullptr;

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);

    return kernelNode;
}
cudaGraphNode_t AppendPlusInplaceBatchNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& A, Tensor& B,
    std::size_t batch) {
    
    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(B.col, BLOCKSIZE), ceil(B.row, BLOCKSIZE));

    void* kernelArgs[] = { &A.data, &B.data, &A.pitch, &B.pitch, &batch, &B.row, &B.col };

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)PlusInplaceBatchKernel;
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = nullptr;

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);

    return kernelNode;
}
cudaGraphNode_t AppendPlusReduceInplceBatchNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& A, Tensor& B,
    std::size_t batch) {
    
    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(A.col, BLOCKSIZE), ceil(A.row, BLOCKSIZE));

    std::size_t* batchPtr = new std::size_t(batch);
    void* kernelArgs[] = { &A.data, &B.data, &A.pitch, &B.pitch, batchPtr, &A.row, &A.col};

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)PlusReduceInplaceBatchKernel;
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = nullptr;

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);

    return kernelNode;
}
cudaGraphNode_t AppendPlusProductReduceInplceBatchNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& A, Tensor& B, Tensor& C,
    std::size_t batch) {
    
    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(A.col, BLOCKSIZE), ceil(A.row, BLOCKSIZE));

    std::size_t* batchPtr = new std::size_t(batch);
    void* kernelArgs[] = { &A.data, &B.data, &C.data, &A.pitch, &B.pitch, &C.pitch, batchPtr, &A.row, &A.col};

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)PlusProductReduceInplaceBatchKernel;
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = nullptr;

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);

    return kernelNode;
}


cudaGraphNode_t AppendMulNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& A, const float x, Tensor& C) {

    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(C.col, BLOCKSIZE), ceil(C.row, BLOCKSIZE));

    float* xPtr = new float(x);
    void* kernelArgs[] = { &A.data, xPtr, &C.data, &A.pitch, &C.pitch, &A.row, &A.col};

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)static_cast<void(*)(
                                            const float*, const float, float*,
                                            const std::size_t, const std::size_t,
                                            const std::size_t, const std::size_t)>
                                            (MulKernel);
    kernelParams.kernelParams = kernelArgs;
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = 0;
    kernelParams.extra = nullptr;

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);

    return kernelNode;
}
cudaGraphNode_t AppendMulInplaceNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& A, const float x) {

    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(A.col, BLOCKSIZE), ceil(A.row, BLOCKSIZE));

    float* xPtr = new float(x);
    void* kernelArgs[] = { &A.data, xPtr, &A.pitch, &A.row, &A.col};

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)static_cast<void(*)(
                                            float*, const float,
                                            const std::size_t,
                                            const std::size_t, const std::size_t)>
                                            (MulInplaceKernel);
    kernelParams.kernelParams = kernelArgs;
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = 0;
    kernelParams.extra = nullptr;

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);

    return kernelNode;
}

cudaGraphNode_t AppendDivNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& A, const float x, Tensor& C) {
    
    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(C.col, BLOCKSIZE), ceil(C.row, BLOCKSIZE));

    float* xInvPtr = new float(1.0f / x);
    void* kernelArgs[] = { &A.data, xInvPtr, &C.data, &A.pitch, &C.pitch, &A.row, &A.col};

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)static_cast<void(*)(
                                            const float*, const float, float*,
                                            const std::size_t, const std::size_t,
                                            const std::size_t, const std::size_t)>
                                            (MulKernel);
    kernelParams.kernelParams = kernelArgs;
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = 0;
    kernelParams.extra = nullptr;

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);

    return kernelNode;
}
cudaGraphNode_t AppendDivInplaceNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& A, const float x) {

    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(A.col, BLOCKSIZE), ceil(A.row, BLOCKSIZE));

    float* xInvPtr = new float(1.0f / x);
    void* kernelArgs[] = { &A.data, xInvPtr, &A.pitch, &A.row, &A.col};

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)static_cast<void(*)(
                                            float*, const float,
                                            const std::size_t,
                                            const std::size_t, const std::size_t)>
                                            (MulInplaceKernel);
    kernelParams.kernelParams = kernelArgs;
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = 0;
    kernelParams.extra = nullptr;

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);

    return kernelNode;
}

cudaGraphNode_t AppendResetNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& A) {

    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    cudaGraphNode_t memsetNode;
    cudaMemsetParams memsetParams = {};
    memsetParams.dst = A.data;
    memsetParams.value = 0;
    memsetParams.pitch = A.pitch;
    memsetParams.elementSize = sizeof(float);
    memsetParams.width = A.col;
    memsetParams.height = A.row;

    cudaGraphAddMemsetNode(&memsetNode, graph, &dependency, numDependency, &memsetParams);

    return memsetNode;
}


cudaGraphNode_t AppendReduceSumOfProductNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& A, Tensor& B, Tensor& sumOfProduct) {
    
    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    dim3 blockDim(REDUCTION_BLOCKSIZE_X, REDUCTION_BLOCKSIZE_Y);
    dim3 gridDim(ceil(1, REDUCTION_BLOCKSIZE_X), ceil(A.row, REDUCTION_BLOCKSIZE_Y));

    void* kernelArgs[] = { &A.data, &B.data, &sumOfProduct.data, &A.pitch, &B.pitch, &A.row, &A.col };

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)ReduceSumOfProductKernel;
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = nullptr;

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);

    return kernelNode;
}
cudaGraphNode_t AppendReduceSumNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& input, Tensor& sum) {
    
    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    dim3 blockDim(REDUCTION_BLOCKSIZE_X, REDUCTION_BLOCKSIZE_Y);
    dim3 gridDim(ceil(1, REDUCTION_BLOCKSIZE_X), ceil(input.row, REDUCTION_BLOCKSIZE_Y));

    void* kernelArgs[] = { &input.data, &sum.data, &input.pitch, &input.row, &input.col };

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)ReduceSumKernel;
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = nullptr;

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);

    return kernelNode;
}
cudaGraphNode_t AppendReduceMaxNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& input, Tensor& maxValue) {
    
    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    dim3 blockDim(REDUCTION_BLOCKSIZE_X, REDUCTION_BLOCKSIZE_Y);
    dim3 gridDim(ceil(1, REDUCTION_BLOCKSIZE_X), ceil(input.row, REDUCTION_BLOCKSIZE_Y));

    void* kernelArgs[] = { &input.data, &maxValue.data, &input.pitch, &input.row, &input.col };

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)ReduceMaxKernel;
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = nullptr;

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);

    return kernelNode;
}


cudaGraphNode_t AppendMeanNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& input, Tensor& mean) {
    
    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    dim3 blockDim(REDUCTION_BLOCKSIZE_X, REDUCTION_BLOCKSIZE_Y);
    dim3 gridDim(ceil(1, REDUCTION_BLOCKSIZE_X), ceil(input.row, REDUCTION_BLOCKSIZE_Y));

    void* kernelArgs[] = { &input.data, &mean.data, &input.pitch, &input.row, &input.col };

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)MeanKernel;
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = nullptr;

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);

    return kernelNode;
}
cudaGraphNode_t AppendStdNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& input, Tensor& mean, Tensor& std) {
    
    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    dim3 blockDim(REDUCTION_BLOCKSIZE_X, REDUCTION_BLOCKSIZE_Y);
    dim3 gridDim(ceil(1, REDUCTION_BLOCKSIZE_X), ceil(input.row, REDUCTION_BLOCKSIZE_Y));

    void* kernelArgs[] = { &input.data, &mean.data, &std.data, &input.pitch, &input.row, &input.col };

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)StdKernel;
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = nullptr;

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);

    return kernelNode;
}


cudaGraphNode_t AppendLookAheadMaskBatchNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& Attention, const std::size_t batch, std::size_t* seq, const float x) {

    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(Attention.col, BLOCKSIZE), ceil(Attention.row, BLOCKSIZE));

    std::size_t* batchPtr = new std::size_t(batch);
    float* xPtr = new float(x);
    void* kernelArgs[] = { &Attention.data, &seq, xPtr, &Attention.pitch, batchPtr, &Attention.row, &Attention.col };

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)ApplyLookAheadMaskBatchKernel;
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = nullptr;

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);

    return kernelNode;
}
cudaGraphNode_t AppendPaddingMaskBatchNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& Attention, const std::size_t batch, std::size_t* seq, const float x) {

    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(Attention.col, BLOCKSIZE), ceil(Attention.row, BLOCKSIZE));

    std::size_t* batchPtr = new std::size_t(batch);
    float* xPtr = new float(x);
    void* kernelArgs[] = { &Attention.data, &seq, xPtr, &Attention.pitch, batchPtr, &Attention.row, &Attention.col };

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)ApplyPaddingMaskBatchKernel;
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = nullptr;

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);

    return kernelNode;
}
cudaGraphNode_t AppendCrossPaddingMaskBatchNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& Attention, const std::size_t batch, std::size_t* seq, const float x) {

    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(Attention.col, BLOCKSIZE), ceil(Attention.row, BLOCKSIZE));

    std::size_t* batchPtr = new std::size_t(batch);
    float* xPtr = new float(x);
    void* kernelArgs[] = { &Attention.data, &seq, xPtr, &Attention.pitch, batchPtr, &Attention.row, &Attention.col };

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)ApplyCrossPaddingMaskBatchKernel;
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = nullptr;

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);

    return kernelNode;
}


cudaGraphNode_t AppendMatMulPlusNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& A, Tensor& B, Tensor& C, bool ATransposed, bool BTransposed) {
    
    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    dim3 blockDim(MATMUL_BLOCKSIZE, MATMUL_BLOCKSIZE);
    dim3 gridDim(ceil(C.col, MATMUL_BLOCKSIZE), ceil(C.row, MATMUL_BLOCKSIZE));

    cudaKernelNodeParams kernelParams = {};
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = 0;
    kernelParams.extra = nullptr;

    if(!ATransposed && !BTransposed) {
        void* kernelArgs[] = { &A.data, &B.data, &C.data, &A.pitch, &B.pitch, &C.pitch, &C.row, &A.col, &C.col};
        kernelParams.func = (void*)MatMulKernelAB;
        kernelParams.kernelParams = kernelArgs;
    }
    else if(ATransposed && !BTransposed) {
        void* kernelArgs[] = { &A.data, &B.data, &C.data, &A.pitch, &B.pitch, &C.pitch, &C.row, &A.row, &C.col};
        kernelParams.func = (void*)MatMulKernelATB;
        kernelParams.kernelParams = kernelArgs;
    }
    else if (!ATransposed && BTransposed) {
        void* kernelArgs[] = { &A.data, &B.data, &C.data, &A.pitch, &B.pitch, &C.pitch, &C.row, &A.col, &C.col};
        kernelParams.func = (void*)MatMulKernelABT;
        kernelParams.kernelParams = kernelArgs;
    }
    else {
        // nothing implemented here.
    }

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);

    return kernelNode;
}

cudaGraphNode_t AppendMatMulPlusBatchNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& A, Tensor& B, Tensor& C, const bool ATransposed, const bool BTransposed,
    const std::size_t batch, const bool ABroadcast, const bool BBroadcast, const bool CBroadcast) {

    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    std::size_t* Asr = new std::size_t(A.row / batch);
    std::size_t* Bsr = new std::size_t(B.row / batch);
    std::size_t* Csr = new std::size_t(C.row / batch);
    dim3 blockDim(MATMUL_BLOCKSIZE, MATMUL_BLOCKSIZE);
    dim3 gridDim(ceil(C.col, MATMUL_BLOCKSIZE), ceil(CBroadcast ? C.row : *Csr, MATMUL_BLOCKSIZE));

    std::vector<cudaGraphNode_t> nodes(batch);

    for(std::size_t i = 0;i < batch;i++) {

        cudaKernelNodeParams kernelParams = {};
        kernelParams.gridDim = gridDim;
        kernelParams.blockDim = blockDim;
        kernelParams.sharedMemBytes = 0;
        kernelParams.extra = nullptr;

        if(!ATransposed && !BTransposed) {
            float** APtr = new float*(ABroadcast ? A.data : GetRow(A.data, *Asr * i, A.pitch));
            float** BPtr = new float*(BBroadcast ? B.data : GetRow(B.data, *Bsr * i, B.pitch));
            float** CPtr = new float*(CBroadcast ? C.data : GetRow(C.data, *Csr * i, C.pitch));
            void* kernelArgs[] = { 
                APtr, BPtr, CPtr,
                &A.pitch, &B.pitch, &C.pitch, 
                CBroadcast ? &C.row : Csr, &A.col, &C.col};
            kernelParams.func = (void*)MatMulKernelAB;
            kernelParams.kernelParams = kernelArgs;
        }
        else if(ATransposed && !BTransposed) {
            float** APtr = new float*(ABroadcast ? A.data : GetRow(A.data, *Asr * i, A.pitch));
            float** BPtr = new float*(BBroadcast ? B.data : GetRow(B.data, *Bsr * i, B.pitch));
            float** CPtr = new float*(CBroadcast ? C.data : GetRow(C.data, *Csr * i, C.pitch));
            void* kernelArgs[] = { 
                APtr, BPtr, CPtr,
                &A.pitch, &B.pitch, &C.pitch, 
                CBroadcast ? &C.row : Csr, ABroadcast ? &A.row : Asr, &C.col};
            kernelParams.func = (void*)MatMulKernelATB;
            kernelParams.kernelParams = kernelArgs;
        }
        else if (!ATransposed && BTransposed) {
            float** APtr = new float*(ABroadcast ? A.data : GetRow(A.data, *Asr * i, A.pitch));
            float** BPtr = new float*(BBroadcast ? B.data : GetRow(B.data, *Bsr * i, B.pitch));
            float** CPtr = new float*(CBroadcast ? C.data : GetRow(C.data, *Csr * i, C.pitch));
            void* kernelArgs[] = { 
                APtr, BPtr, CPtr,
                &A.pitch, &B.pitch, &C.pitch, 
                CBroadcast ? &C.row : Csr, &A.col, &C.col};
            kernelParams.func = (void*)MatMulKernelABT;
            kernelParams.kernelParams = kernelArgs;
        }
        else {
            // nothing implemented here.
        }

        cudaGraphNode_t kernelNode;
        cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);
        nodes[i] = kernelNode;

        if(CBroadcast) {
            dependency = nodes[i];
            numDependency = 1;
        }
    }

    return CBroadcast ? nodes.back() : SyncDependency(graph, nodes);
}