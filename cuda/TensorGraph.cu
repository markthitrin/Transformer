#include "Header.cuh"
#include "Tensor.cuh"
#include "TensorKernel.cuh"
#include "Util.cuh"
#include "UtilKernel.cuh"

cudaGraphNode_t AppendCopyBatchNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor A, Tensor B, const std::size_t batch) {
    
    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);

    const int sr = B.row / batch;
    std::vector<cudaGraphNode_t> nodes(batch);
    for(int i = 0;i < batch;i++) {
        cudaMemcpy3DParms params = {0};
        params.srcPtr = make_cudaPitchedPtr(A.data, A.pitch, sizeof(float) * A.col, A.row);
        params.dstPtr = make_cudaPitchedPtr(Get(B.data, i * sr, 0, B.pitch), B.pitch, sizeof(float) * A.col, A.row);
        params.extent = make_cudaExtent(sizeof(float) * A.col, A.row, 1);
        params.kind = cudaMemcpyDeviceToDevice;
        cudaGraphAddMemcpyNode(&nodes[i], graph, &dependency, 1, &params);
    }

    return SyncDependency(graph, nodes);
}


cudaGraphNode_t AppendPlusBatchNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor A, Tensor B, Tensor C,
    std::size_t batch) {
    
    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);

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
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, 1, &kernelParams);

    return kernelNode;
}
cudaGraphNode_t AppendPlusInplceBatchNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor A, Tensor C,
    std::size_t batch) {
    
    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);

    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(C.col, BLOCKSIZE), ceil(C.row, BLOCKSIZE));

    void* kernelArgs[] = { &A.data, &C.data, &A.pitch, &C.pitch, &batch, &C.row, &C.col};

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)PlusInplaceBatchKernel;
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = nullptr;

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, 1, &kernelParams);

    return kernelNode;
}


cudaGraphNode_t AppendMulNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor A, const float x, Tensor C) {

    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    dim3 blockDim(MATMUL_BLOCKSIZE, MATMUL_BLOCKSIZE);
    dim3 gridDim(ceil(C.col, MATMUL_BLOCKSIZE), ceil(C.row, MATMUL_BLOCKSIZE));

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
    Tensor A, const float x) {

    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    dim3 blockDim(MATMUL_BLOCKSIZE, MATMUL_BLOCKSIZE);
    dim3 gridDim(ceil(A.col, MATMUL_BLOCKSIZE), ceil(A.row, MATMUL_BLOCKSIZE));

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


cudaGraphNode_t AppendResetNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor A) {

    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);

    cudaGraphNode_t memsetNode;
    cudaMemsetParams memsetParams = {};
    memsetParams.dst = A.data;
    memsetParams.value = 0;
    memsetParams.pitch = A.pitch;
    memsetParams.elementSize = sizeof(float);
    memsetParams.width = A.col;
    memsetParams.height = A.row;

    cudaGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams);

    return memsetNode;
}



cudaGraphNode_t AppendMatMulPlus(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor A, Tensor B, Tensor C, bool ATransposed, bool BTransposed) {
    
    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);

    dim3 blockDim(MATMUL_BLOCKSIZE, MATMUL_BLOCKSIZE);
    dim3 gridDim(ceil(C.col, MATMUL_BLOCKSIZE), ceil(C.row, MATMUL_BLOCKSIZE));

    cudaKernelNodeParams kernelParams = {};
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = MATMUL_BLOCKSIZE * MATMUL_BLOCKSIZE * sizeof(float) * 2;
    kernelParams.extra = nullptr;

    if(!ATransposed && !BTransposed) {
        void* kernelArgs[] = { &A.data, &B.data, &C.data, &A.pitch, &B.pitch, &C.pitch, &C.row, &A.col, &B.col};
        kernelParams.func = (void*)MatMulKernelAB;
        kernelParams.kernelParams = kernelArgs;
    }
    else if(ATransposed && !BTransposed) {
        void* kernelArgs[] = { &A.data, &B.data, &C.data, &A.pitch, &B.pitch, &C.pitch, &C.row, &A.row, &C.col};
        kernelParams.func = (void*)MatMulKernelAB;
        kernelParams.kernelParams = kernelArgs;
    }
    else if (!ATransposed && BTransposed) {
        void* kernelArgs[] = { &A.data, &B.data, &C.data, &A.pitch, &B.pitch, &C.pitch, &C.row, &A.col, &C.col};
        kernelParams.func = (void*)MatMulKernelAB;
        kernelParams.kernelParams = kernelArgs;
    }
    else {
        // nothing implemented here.
    }

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, 1, &kernelParams);

    return kernelNode;
}