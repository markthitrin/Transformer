#include "Header.cuh"
#include "Tensor.cuh"
#include "TensorFunction.cuh"
#include "TensorKernel.cuh"
#include "TensorGraph.cuh"
#include "Util.cuh"
#include "UtilKernel.cuh"

AdamOptimizer::AdamOptimizer(const Tensor& param) : 
    gradient(param.row, param.col),
    accM(param.row, param.col),
    accV(param.row, param.col) {
    
    cudaMalloc(&t, sizeof(std::size_t));
    Reset(gradient);
    Reset(accM);
    Reset(accV);
    std::size_t _t = 1;
    cudaMemcpy(t, &_t, sizeof(std::size_t), cudaMemcpyHostToDevice);
}

AdamOptimizer::AdamOptimizer(const std::size_t row, const std::size_t col) : 
    gradient(row, col),
    accM(row, col),
    accV(row, col) {
    
    cudaMalloc(&t, sizeof(std::size_t));
    Reset(gradient);
    Reset(accM);
    Reset(accV);
    std::size_t _t = 1;
    cudaMemcpy(t, &_t, sizeof(std::size_t),cudaMemcpyHostToDevice);
}

AdamOptimizer::AdamOptimizer(const AdamOptimizer& other) :
    gradient(other.gradient),
    accM(other.accM),
    accV(other.accV),
    t(other.t) {;}

AdamOptimizer::AdamOptimizer(AdamOptimizer&& other) :
    gradient(std::move(other.gradient)),
    accM(std::move(other.accM)),
    accV(std::move(other.accV)),
    t(std::move(other.t)) {;}

AdamOptimizer::~AdamOptimizer() {
    cudaFree(t);
}





void AdamOpt(Tensor& param, AdamOptimizer opt) {
    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(param.col, BLOCKSIZE), ceil(param.row, BLOCKSIZE));
    AdamOptKernel<<<gridDim, blockDim>>>(
        param.data, opt.gradient.data, opt.accM.data, opt.accV.data, opt.t,
        param.pitch, opt.gradient.pitch, opt.accM.pitch, opt.accV.pitch,
        param.row, param.col);
}
cudaGraphNode_t AppendAdamOptNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& param, AdamOptimizer& opt) {
    
    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(param.col, BLOCKSIZE), ceil(param.row, BLOCKSIZE));

    void* kernelArgs[] = { 
        &param.data, &opt.gradient.data, &opt.accM.data, &opt.accV.data, &opt.t,
        &param.pitch, &opt.gradient.pitch, &opt.accM.pitch, &opt.accV.pitch,
        &param.row, &param.col
    };

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)AdamOptKernel;
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = nullptr;

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);

    return kernelNode;
}

float CrossEntropy(Tensor& logits, const std::size_t* targetH, Tensor& gradient, std::size_t* tgtSeqH) {
    Reset(gradient);

    std::size_t sumCase = 0;

    static float* lossH = new float[logits.row];
    static bool* tgtSeqHotH = new bool[logits.row];
    
    static Tensor loss(1, logits.row);
    static Tensor maxValue(1, logits.row);
    static Tensor sumExp(1, logits.row);
    static bool init = false; 
    std::size_t* target = nullptr;
    bool* tgtSeqHot = nullptr;
    if(!init) {
        cudaMalloc(&target, sizeof(std::size_t) * logits.row);
        cudaMalloc(&tgtSeqHot, sizeof(bool) * logits.row);
    }
    for(int i = 0;i < batch;i++) {
        sumCase += tgtSeqH[i];
        for(int j = 0;j < sequenceLength;j++) {
            tgtSeqHotH[i * sequenceLength + j] = j < tgtSeqH[i];
        }
    }
    cudaMemcpy(target, targetH, sizeof(std::size_t) * logits.row, cudaMemcpyHostToDevice);
    cudaMemcpy(tgtSeqHot, tgtSeqHotH, sizeof(bool) * logits.row, cudaMemcpyHostToDevice);

    ReduceMax(logits, maxValue);
    ReduceSumExp(logits, maxValue, sumExp);
    SoftmaxF(logits, sumExp, maxValue, gradient, tgtSeqHot);
    CrossEntropyF(logits, sumExp, maxValue, gradient, target, tgtSeqHot, loss);
    cudaDeviceSynchronize();
    
    loss.toFloat(lossH);
    float totalLoss = 0.0f;
    for(int i = 0;i < logits.row;i++) {
        totalLoss += lossH[i];
    }
    
    if(sumCase != 0) totalLoss /= sumCase;
    return totalLoss;
}

void Print(Tensor& A, const std::size_t r0, const std::size_t c0, const std::size_t r, const std::size_t c) {
    float* _A = (float*)malloc(sizeof(float) * A.row * A.col);
	A.toFloat(_A);

    for(int i = r0;i < r0 + r;i++) {
        for(int j = c0;j < c0 + c;j++) {
            std::cout << _A[i * A.col + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    delete[] _A;
}

cudaGraphNode_t SyncDependency(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    if(dependencyNodes.size() > 1) {
        cudaGraphNode_t dependency = nullptr;
        cudaGraphAddEmptyNode(&dependency, graph, dependencyNodes.data(), dependencyNodes.size());
        return dependency;
    }
    else if(dependencyNodes.size()) {
        return dependencyNodes[0];
    }
    return nullptr;
}

void CrossEntropyF(Tensor& logits, Tensor& sumExp, Tensor& maxValue, Tensor& gradient, const std::size_t* target, const bool* tgtSeqHot, Tensor& loss) {
    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE);
    dim3 gridDim(ceil(logits.row, BLOCKSIZE));
    CrossEntropyKernel<<<gridDim, blockDim>>>(
        logits.data, sumExp.data, maxValue.data, gradient.data, target, tgtSeqHot, loss.data,
        logits.pitch, gradient.pitch,
        logits.row); 
}

void SoftmaxF(Tensor& input, Tensor& sumExp, Tensor& maxValue, Tensor& output, const bool* tgtSeqHot) {
    const int BLOCK_SIZE = 16;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(ceil(input.col, BLOCK_SIZE), ceil(input.row, BLOCK_SIZE));
    SoftmaxFKernel<<<gridDim, blockDim>>>(input.data, maxValue.data, sumExp.data, output.data, tgtSeqHot,
        input.pitch, output.pitch,
        input.row, input.col);
}

