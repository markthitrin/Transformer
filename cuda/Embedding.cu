#include "Header.cuh"
#include "Tensor.cuh"
#include "TensorFunction.cuh"
#include "TensorKernel.cuh"
#include "TensorGraph.cuh"
#include "Util.cuh"
#include "UtilKernel.cuh"
#include "Embedding.cuh"

Embedding::Embedding(
    std::size_t* input,
    Tensor& output,
    Tensor& outputGradient,
    const std::size_t token) noexcept :

    input(input),
    output(output),
    outputGradient(outputGradient),
    
    entries(batch * sequenceLength, nullptr),
    forwardNodes(batch * sequenceLength, nullptr),
    forwardNodeParams(batch * sequenceLength),
    backwardNodes(batch * sequenceLength, nullptr),
    backwardNodeParams(batch * sequenceLength),
    updateParameterNodes(batch * sequenceLength, nullptr),
    updateParameterParams(batch * sequenceLength) {

    table.reserve(token);
    tableOpt.reserve(token);
    for(int i = 0;i < token;i++) {
        table.emplace_back(1, dModel);
        table[i].UniformFill(0.1f);

        tableOpt.emplace_back(1, dModel);
    }

    for(int i = 0;i < batch * sequenceLength;i++) {
        entries[i] = GetRow(outputGradient.data, i, outputGradient.pitch);
    }
}
Embedding::~Embedding() {
    for(int i = 0;i < table.size();i++) {
        table[i].free();
    }
}

void Embedding::UpdateGraph(cudaGraphExec_t graphExec) {
    std::set<std::size_t> ss;
    for(std::size_t i = 0;i < batch * sequenceLength;i++) {
        ss.insert(input[i]);

        forwardNodeParams[i].srcPtr = make_cudaPitchedPtr(
            table[input[i]].data, table[input[i]].pitch,
            table[input[i]].col, table[input[i]].row);
        cudaError_t err =cudaGraphExecMemcpyNodeSetParams(graphExec, forwardNodes[i], &forwardNodeParams[i]);
        PRINT_CUDA_ERR(err);
        
        void* kernelArgsBackward[] = { 
            &tableOpt[input[i]].gradient, &entries[i], 
            &tableOpt[input[i]].gradient.pitch, &outputGradient.pitch, 
            &tableOpt[input[i]].gradient.row, &tableOpt[input[i]].gradient.col}; 
        backwardNodeParams[i].kernelParams = kernelArgsBackward;
        cudaGraphExecKernelNodeSetParams(graphExec, backwardNodes[i], &backwardNodeParams[i]);
    }
    for(int i = 0;i < batch * sequenceLength;i++) {
        
        if(ss.count(input[i]) != 0) { // working node
            // void* kernelArgsUpdate[] = {
            //     &table[input[i]].data, &tableOpt[input[i]].gradient.data, &tableOpt[input[i]].accM.data, &tableOpt[input[i]].accV.data, &tableOpt[input[i]].t,
            //     &table[input[i]].pitch, &tableOpt[input[i]].gradient.pitch, &tableOpt[input[i]].accM.pitch, &tableOpt[input[i]].accV.pitch,
            //     &table[input[i]].row, &table[input[i]].col};

            // constexpr int BLOCKSIZE = 16;
            // dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
            // dim3 gridDim(ceil(table[input[i]].col, BLOCKSIZE), ceil(table[input[i]].row, BLOCKSIZE));
            
            // updateParameterParams[i].gridDim = gridDim;
            // updateParameterParams[i].blockDim = blockDim;
            // updateParameterParams[i].kernelParams = kernelArgsUpdate;

            cudaError_t err = cudaGraphExecKernelNodeSetParams(graphExec, updateParameterNodes[i], &updateParameterParams[i]);
            PRINT_CUDA_ERR(err);

            ss.erase(input[i]);
        }
        else { // empty node
            updateParameterParams[i].gridDim = dim3(0, 0, 0);
            updateParameterParams[i].blockDim = dim3(0, 0, 0);
            cudaError_t err = cudaGraphExecKernelNodeSetParams(graphExec, updateParameterNodes[i], &updateParameterParams[i]);
            PRINT_CUDA_ERR(err);
        }
    }
}

cudaGraphNode_t Embedding::AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = AppendEmbeddingNode(graph, dependencyNodes, forwardNodeParams, forwardNodes, output);
    cudaGraphNode_t k2 = AppendMulInplaceNode(graph, {k1}, output, std::sqrt(dModel));
    return k2;
}

cudaGraphNode_t Embedding::AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    return AppendGraphForward(graph, dependencyNodes);
}

cudaGraphNode_t Embedding::AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = AppendMulInplaceNode(graph, dependencyNodes, outputGradient, std::sqrt(dModel));
    cudaGraphNode_t k2 = AppendEmbeddingBackwardNode(graph, {k1}, backwardNodeParams, backwardNodes, outputGradient);
    return k2;
}

cudaGraphNode_t Embedding::AppendGraphUpdateParameter(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = AppendEmbeddingUpdateParameterNode(graph, dependencyNodes, updateParameterParams, updateParameterNodes, outputGradient);
    return k1;
}

void Embedding::loadParam(cnpy::npz_t npFile, std::string prefix) {
    Tensor loadRR(table.size(), output.col);
    loadRR.loadNp(npFile, prefix + ".weight");
    for(int i = 0;i < table.size();i++) {
        cudaMemcpy2DAsync(
            table[i].data, table[i].pitch, Get(loadRR.data, i, 0, loadRR.pitch), loadRR.pitch,
            sizeof(float) * table[i].col, 1, cudaMemcpyDeviceToDevice);
    }
    cudaDeviceSynchronize();
}

void Embedding::forwardTest(cnpy::npz_t npFile, std::string prefix) {
    Tensor target(output.row, output.col);

    float* temp = new float[batch * sequenceLength];
    Tensor loadInput(1, batch * sequenceLength);
    loadInput.loadNp(npFile, prefix + ".input");
    loadInput.toFloat(temp);
    for(int i = 0;i < batch * sequenceLength;i++) {
        input[i] = (std::size_t)temp[i];
    }
    target.loadNp(npFile, prefix + ".output");

    // Forward
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaGraphCreate(&graph, 0);
    this->AppendGraphForward(graph, {});
    cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
    this->UpdateGraph(instance);
    cudaGraphLaunch(instance, 0);
    cudaDeviceSynchronize();

    PrintTestResult("forward", output, target);
    delete[] temp;
}

void Embedding::checkUpdatedParam(cnpy::npz_t npFile, std::string prefix) {
    std::vector<Tensor> updatedTable;
    for(int i = 0;i < table.size();i++) updatedTable.emplace_back(1, dModel);

    Tensor loadRR(table.size(), table[0].col);
    loadRR.loadNp(npFile, prefix + ".updated_weights");
    for(int i = 0;i < updatedTable.size();i++) {
        cudaMemcpy2D(
            updatedTable[i].data, updatedTable[i].pitch, Get(loadRR.data, i, 0, loadRR.pitch), loadRR.pitch,
            sizeof(float) * updatedTable[i].col, 1, cudaMemcpyDeviceToDevice);
    }
    cudaDeviceSynchronize();

    for(int i = 0;i < updatedTable.size();i++) {
        PrintTestResult("backward table:" + std::to_string(i), table[i], updatedTable[i]);
    }
}

void Embedding::backwardTest(cnpy::npz_t npFile, std::string prefix) {
    Set(outputGradient, 1.0f / output.row / output.col);
    cudaDeviceSynchronize();

    // load input
    float* temp = new float[batch * sequenceLength];
    Tensor loadInput(1, batch * sequenceLength);
    loadInput.loadNp(npFile, prefix + ".input");
    loadInput.toFloat(temp);
    for(int i = 0;i < batch * sequenceLength;i++) {
        input[i] = (std::size_t)temp[i];
    }

    // Forward, backward, update
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaGraphCreate(&graph, 0);
    cudaGraphNode_t k1 = this->AppendGraphForward(graph, {});
    cudaGraphNode_t k2 = this->AppendGraphBackward(graph, {k1});
    this->AppendGraphUpdateParameter(graph, {k2});
    cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
    this->UpdateGraph(instance);
    cudaGraphLaunch(instance, 0);
    cudaDeviceSynchronize();

    checkUpdatedParam(npFile, prefix);
}


cudaGraphNode_t AppendEmbeddingNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    std::vector<cudaMemcpy3DParms>& forwardNodeParams, std::vector<cudaGraphNode_t>& forwardNodes,
    Tensor& output) {

    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    for(std::size_t i = 0;i < output.row;i++) {
        cudaMemcpy3DParms copyParams = {};
        copyParams.srcArray = nullptr;
        copyParams.dstArray = nullptr;
        copyParams.srcPtr = make_cudaPitchedPtr(output.data, output.pitch, output.col, 1);
        copyParams.dstPtr = make_cudaPitchedPtr(GetRow(output.data, i, output.pitch), output.pitch, output.col, 1);
        copyParams.extent = make_cudaExtent(sizeof(float) * output.col, 1, 1);
        copyParams.kind = cudaMemcpyDeviceToDevice;

        cudaGraphNode_t copyNode;
        cudaError_t err = cudaGraphAddMemcpyNode(&copyNode, graph, &dependency, numDependency, &copyParams);
        PRINT_CUDA_ERR(err);

        forwardNodes[i] = copyNode;
        forwardNodeParams[i] = copyParams;
    }
    
    return SyncDependency(graph, forwardNodes);
}

cudaGraphNode_t AppendEmbeddingBackwardNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    std::vector<cudaKernelNodeParams>& backwardNodeParams, std::vector<cudaGraphNode_t>& backwardNodes,
    Tensor& outputGradient) {

    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(outputGradient.col, BLOCKSIZE), ceil(1, BLOCKSIZE));

    for(std::size_t i = 0;i < batch * sequenceLength;i++) {
        float* dummyFloatPtr = nullptr;
        std::size_t dummySize = 0;

        cudaKernelNodeParams kernelParams = {};
        void* kernelArgs[] = { &dummyFloatPtr, &dummyFloatPtr, &dummySize, &dummySize, &dummySize, &dummySize}; 

        kernelParams.func = (void*)PlusInplaceKernel;
        kernelParams.gridDim = gridDim;
        kernelParams.blockDim = blockDim;
        kernelParams.sharedMemBytes = 0;
        kernelParams.kernelParams = kernelArgs;
        kernelParams.extra = nullptr;

        cudaGraphNode_t kernelNode;
        cudaError_t err = cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);
        PRINT_CUDA_ERR(err);

        backwardNodes[i] = kernelNode;
        backwardNodeParams[i] = kernelParams;

        dependency = backwardNodes[i];
        numDependency = 1;
    }

    return backwardNodes.back();
}

cudaGraphNode_t AppendEmbeddingUpdateParameterNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    std::vector<cudaKernelNodeParams>& updateParameterParams, std::vector<cudaGraphNode_t>& updateParameterNodes,
    Tensor& outputGradient) {
    
    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(outputGradient.col, BLOCKSIZE), ceil(1, BLOCKSIZE));

    for(int i = 0;i < batch * sequenceLength;i++) {
        float* dummyFloatPtr = nullptr;
        std::size_t* dummySizePtr = nullptr;
        std::size_t dummySize = 0;

        cudaKernelNodeParams kernelParams;
        void* kernelArgs[] = {
            &dummyFloatPtr, &dummyFloatPtr, &dummyFloatPtr, &dummyFloatPtr, &dummySizePtr,
            &dummySize, &dummySize, &dummySize, &dummySize,
            &dummySize, &dummySize}; 

        kernelParams.func = (void*)AdamOptKernel;
        kernelParams.gridDim = gridDim;
        kernelParams.blockDim = blockDim;
        kernelParams.sharedMemBytes = 0;
        kernelParams.kernelParams = kernelArgs;
        kernelParams.extra = nullptr;

        cudaGraphNode_t kernelNode;
        cudaError_t err = cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);
        PRINT_CUDA_ERR(err);

        updateParameterNodes[i] = kernelNode;
        updateParameterParams[i] = kernelParams;
    }

    return SyncDependency(graph, updateParameterNodes);
}
