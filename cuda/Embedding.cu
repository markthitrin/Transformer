#include "Header.cuh"
#include "Tensor.cuh"
#include "TensorFunction.cuh"
#include "TensorKernel.cuh"
#include "TensorGraph.cuh"
#include "Util.cuh"
#include "UtilKernel.cuh"
#include "Embedding.cuh"

Embedding::Embedding(
    std::size_t*& inputH,
    Tensor& output,
    Tensor& outputGradient,
    const std::size_t numToken) noexcept :

    inputH(inputH),
    output(output),
    outputGradient(outputGradient),
    numToken(numToken),
    
    table(numToken, dModel),
    tableOpt(numToken, dModel),
    
    forwardNodes(batch * sequenceLength),
    forwardParams(batch * sequenceLength) {

    table.UniformFill(0.1f);

    cudaMalloc(&t, sizeof(std::size_t) * numToken);
    cudaMalloc(&input, sizeof(std::size_t) * sequenceLength * batch);

    std::vector<std::size_t> _t(numToken, 1);
    cudaMemcpy(t, _t.data(), sizeof(std::size_t) * numToken, cudaMemcpyHostToDevice);
}
Embedding::~Embedding() {
    cudaFree(t);
    cudaFree(input);
}

void Embedding::UpdateGraph(cudaGraphExec_t instance) {
	cudaMemcpy(input, inputH, sizeof(std::size_t) * sequenceLength * batch,  cudaMemcpyHostToDevice);

    for(std::size_t i = 0;i < batch * sequenceLength;i++) {
        std::size_t inputToken = inputH[i];
        forwardParams[i].srcPtr = make_cudaPitchedPtr(
            GetRow(table.data, inputToken, table.pitch), table.pitch,
            table.col, 1);
        cudaError_t err = cudaGraphExecMemcpyNodeSetParams(instance, forwardNodes[i], &forwardParams[i]);
        PRINT_CUDA_ERR(err);
        if(err != cudaSuccess) {
            std::cout << "df";
        }
    }
}

cudaGraphNode_t Embedding::AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = AppendEmbeddingNode(graph, dependencyNodes, input, forwardNodes, forwardParams, table, output);
    cudaGraphNode_t k2 = AppendMulInplaceNode(graph, {k1}, output, std::sqrt(dModel));
    return k2;
}

cudaGraphNode_t Embedding::AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    return AppendGraphForward(graph, dependencyNodes);
}

cudaGraphNode_t Embedding::AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = AppendMulInplaceNode(graph, dependencyNodes, outputGradient, std::sqrt(dModel));
    cudaGraphNode_t k2 = AppendEmbeddingBackwardNode(graph, {k1}, input, tableOpt, outputGradient);
    return k2;
}

cudaGraphNode_t Embedding::AppendGraphUpdateParameter(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    cudaGraphNode_t k1 = AppendEmbeddingUpdateParameterNode(graph, dependencyNodes, input, table, tableOpt, t, outputGradient);
    return k1;
}

void Embedding::loadParam(cnpy::npz_t npFile, std::string prefix) {
    table.loadNp(npFile, prefix + ".weight");
}

void Embedding::forwardTest(cnpy::npz_t npFile, std::string prefix) {
    Tensor target(output.row, output.col);
    Tensor inputLoader(1, batch * sequenceLength);

    target.loadNp(npFile, prefix + ".output");
    inputLoader.loadNp(npFile, prefix + ".input");
    float* _inputH = new float[batch * sequenceLength];
    inputLoader.toFloat(_inputH);
    for(int i = 0;i < batch * sequenceLength;i++) inputH[i] = _inputH[i];

    // Forward
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaGraphCreate(&graph, 0);
    this->AppendGraphForward(graph, {});
    cudaGraphDebugDotPrint(graph, "graph.dot", cudaGraphDebugDotFlagsVerbose);
    cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
    this->UpdateGraph(instance);
    cudaGraphLaunch(instance, 0);
    cudaDeviceSynchronize();

    PrintTestResult("forward", output, target);
}

void Embedding::checkUpdatedParam(cnpy::npz_t npFile, std::string prefix) {
    Tensor targetTable(table.row, table.col);
    targetTable.loadNp(npFile, prefix + ".updated_weights");

    PrintTestResult("backward table " ,table, targetTable);
}

void Embedding::backwardTest(cnpy::npz_t npFile, std::string prefix) {
    Set(outputGradient, 1.0f / output.row / output.col);
    
    Tensor inputLoader(1, batch * sequenceLength);

    inputLoader.loadNp(npFile, prefix + ".input");
    float* _inputH = new float[batch * sequenceLength];
    inputLoader.toFloat(_inputH);
    for(int i = 0;i < batch * sequenceLength;i++) inputH[i] = _inputH[i];

    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaGraphCreate(&graph, 0);
    cudaGraphNode_t k1 = this->AppendGraphForward(graph, {});
    cudaGraphNode_t k2 = this->AppendGraphBackward(graph, {k1});
    this->AppendGraphUpdateParameter(graph, {k2});
    cudaGraphDebugDotPrint(graph, "graph.dot", cudaGraphDebugDotFlagsVerbose);
    cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
    this->UpdateGraph(instance);
    cudaGraphLaunch(instance, 0);
    cudaDeviceSynchronize();

    checkUpdatedParam(npFile, prefix);
}


cudaGraphNode_t AppendEmbeddingNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    std::size_t* input, std::vector<cudaGraphNode_t>& forwardNodes, std::vector<cudaMemcpy3DParms>& forwardParams,
    Tensor& table, Tensor& output) {

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
        forwardParams[i] = copyParams;
    }
    
    return SyncDependency(graph, forwardNodes);
}

cudaGraphNode_t AppendEmbeddingBackwardNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    std::size_t* input,
    AdamOptimizer& tableOpt, Tensor& outputGradient) {

    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    constexpr int BLOCKSIZE = 32;
    dim3 blockDim(BLOCKSIZE);
    dim3 gridDim(ceil(outputGradient.col, BLOCKSIZE));

    float** outputGradientEntries = new float*[batch * sequenceLength];
    std::size_t** inputPtr = new std::size_t*[batch * sequenceLength];
    for(int i = 0 ;i < batch * sequenceLength;i++) {
        outputGradientEntries[i] = GetRow(outputGradient.data, i, outputGradient.pitch);
        inputPtr[i] = input + i;
    }
    
    std::vector<cudaGraphNode_t> nodes(batch * sequenceLength);
    for(std::size_t i = 0;i < batch * sequenceLength;i++) {

        cudaKernelNodeParams kernelParams = {};
        void* kernelArgs[] = { &outputGradientEntries[i], &inputPtr[i], &tableOpt.gradient.data, &tableOpt.gradient.pitch, &outputGradient.col}; 

        kernelParams.func = (void*)EmbeddingBackwardKernel;
        kernelParams.gridDim = gridDim;
        kernelParams.blockDim = blockDim;
        kernelParams.sharedMemBytes = 0;
        kernelParams.kernelParams = kernelArgs;
        kernelParams.extra = nullptr;

        cudaGraphNode_t kernelNode;
        cudaError_t err = cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);
        PRINT_CUDA_ERR(err);
        nodes[i] = kernelNode;

        dependency = nodes[i];
        numDependency = 1;
    }

    return nodes.back();
}

cudaGraphNode_t AppendEmbeddingUpdateParameterNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    std::size_t* input,
    Tensor& table, AdamOptimizer& tableOpt, std::size_t*& t, Tensor& outputGradient) {
    
    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    constexpr int BLOCKSIZE = 32;
    dim3 blockDim(BLOCKSIZE);
    dim3 gridDim(ceil(outputGradient.col, BLOCKSIZE));

    std::size_t** inputPtr = new std::size_t*[batch * sequenceLength];
    for(int i = 0 ;i < batch * sequenceLength;i++) {
        inputPtr[i] = input + i;
    }

    std::vector<cudaGraphNode_t> nodes(batch * sequenceLength);
    for(int i = 0;i < batch * sequenceLength;i++) {

        cudaKernelNodeParams kernelParams;
        void* kernelArgs[] = {
            &inputPtr[i], &table.data, &tableOpt.gradient.data, &tableOpt.accM.data, &tableOpt.accV.data, &t,
            &table.pitch, &tableOpt.gradient.pitch, &tableOpt.accM.pitch, &tableOpt.accV.pitch,
            &table.row, &table.col }; 

        kernelParams.func = (void*)EmbeddingAdamOptKernel;
        kernelParams.gridDim = gridDim;
        kernelParams.blockDim = blockDim;
        kernelParams.sharedMemBytes = 0;
        kernelParams.kernelParams = kernelArgs;
        kernelParams.extra = nullptr;

        cudaGraphNode_t kernelNode;
        cudaError_t err = cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);
        PRINT_CUDA_ERR(err);

        nodes[i] = kernelNode;
        dependency = nodes[i];
        numDependency = 1;
    }

    return nodes.back();
}

__global__ void EmbeddingBackwardKernel(
    const float* outputGradient, const std::size_t* input, float* gradient,
	const std::size_t pitchGradient,
    const std::size_t col) {

    const std::size_t c = blockIdx.x * blockDim.x + threadIdx.x;

    if(c < col) {
        *Get(gradient, *input, c, pitchGradient) += outputGradient[c];
    }
}

__global__ void EmbeddingAdamOptKernel(
    const std::size_t* input, float* param, float* gradient, float* accM, float* accV, std::size_t* t,
    const std::size_t pitchParam, const std::size_t pitchGrad, const std::size_t pitchAccM, const std::size_t pitchAccV,
    const std::size_t row, const std::size_t col) {
   
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = *input;

    if(*Get(gradient, r, c, pitchGrad) != 0) {
        const float invPowBeta1 = __frcp_rn(1.0f - powf(beta1, t[*input]));
        const float invPowBeta2 = __frcp_rn(1.0f - powf(beta2, t[*input]));
        if(r < row && c < col) {
            const float g = *Get(gradient, r, c, pitchGrad);
            *Get(gradient, r, c, pitchGrad) = 0;
            *Get(accM, r, c, pitchAccM) = *Get(accM, r, c, pitchAccM) * beta1 + g * (1.0f - beta1);
            *Get(accV, r, c, pitchAccV) = *Get(accV, r, c, pitchAccV) * beta2 + g * g * (1.0f - beta2);
            float mHat = *Get(accM, r, c, pitchAccM) * invPowBeta1;
            float vHat = *Get(accV, r, c, pitchAccV) * invPowBeta2;
            *Get(param, r, c, pitchParam) -= lr * mHat / (sqrtf(vHat) + eps);
        }
    }
    __syncthreads();
    if(r == 0 && c == 0 && *Get(gradient, r, c, pitchGrad) != 0) {
        t[*input]++;
    }
}
