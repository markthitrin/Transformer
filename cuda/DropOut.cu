#include "Header.cuh"
#include "Tensor.cuh"
#include "DropOut.cuh"
#include "TensorKernel.cuh"
#include "TensorGraph.cuh"
#include "TensorFunction.cuh"
#include "Util.cuh"
#include "UtilKernel.cuh"


DropOut::DropOut(
    Tensor& input,
    Tensor& output,
    Tensor& outputGradient,
    Tensor& inputGradient,
    const std::size_t row,
    const std::size_t col) noexcept:
    
    input(input),
    output(output),
    outputGradient(outputGradient),
    inputGradient(inputGradient),
    mask(row, col) {
        
    cudaMallocPitch((void**)&states, &pitchState, sizeof(float) * col, row);
}

DropOut::~DropOut() {
    cudaFree(states);
}

cudaGraphNode_t DropOut::AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    // cudaGraphNode_t k1 = AppendDropoutNode(graph, dependencyNodes, input, mask, output, states, dropoutRate, pitchState);
    cudaGraphNode_t k1 = AppendMulNode(graph, dependencyNodes, input, 1 / (1.0f - dropoutRate), output);
    return k1;
}

cudaGraphNode_t DropOut::AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    return AppendCopyNode(graph, dependencyNodes, input, output);
}

cudaGraphNode_t DropOut::AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {
    // cudaGraphNode_t k1 = AppendDropoutBackwardNode(graph, dependencyNodes, outputGradient, mask, inputGradient);
    cudaGraphNode_t k1 = AppendMulNode(graph, dependencyNodes, outputGradient, 1 / (1.0f - dropoutRate), inputGradient);
    return k1;
}

cudaGraphNode_t AppendDropoutNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& input, Tensor& mask, Tensor& output, curandStatePhilox4_32_10_t* states, float dropoutRate,
    std::size_t pitchState) {

    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(input.col, BLOCKSIZE), ceil(input.row, BLOCKSIZE));

    void* kernelArgs[] = {
        &input.data, &mask.data, &output.data, &states, &dropoutRate,
        &input.pitch, &mask.pitch, &output.pitch, &pitchState,
        &input.row, &input.col }; 

    cudaKernelNodeParams kernelParams;
    kernelParams.func = (void*)dropoutKernel;
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = nullptr;

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);

    return kernelNode;
}

cudaGraphNode_t AppendDropoutBackwardNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor& outputGradient, Tensor& mask, Tensor& inputGradient) {

    cudaGraphNode_t dependency = SyncDependency(graph, dependencyNodes);
    std::size_t numDependency = dependency == nullptr ? 0 : 1;

    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(outputGradient.col, BLOCKSIZE), ceil(outputGradient.row, BLOCKSIZE));

    void* kernelArgs[] = { 
        &outputGradient, &mask, &inputGradient,
        &outputGradient.pitch, &mask.pitch, &inputGradient.pitch,
        &outputGradient.row, &outputGradient.col}; 

    cudaKernelNodeParams kernelParams = {};
    kernelParams.func = (void*)static_cast<void(*)(
                                const float*, const float*, float*,
                                const std::size_t, const std::size_t, const std::size_t,
                                const std::size_t, const std::size_t)>(MulKernel);
    kernelParams.gridDim = gridDim;
    kernelParams.blockDim = blockDim;
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    kernelParams.extra = nullptr;

    cudaGraphNode_t kernelNode;
    cudaGraphAddKernelNode(&kernelNode, graph, &dependency, numDependency, &kernelParams);

    return kernelNode;
}


__global__ void setupState(
    curandStatePhilox4_32_10_t* states, unsigned long long seed,
    const std::size_t pitchState,
    const std::size_t row, const std::size_t col) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    if(r < row && c < col) {
        curand_init(seed, r * col + c, 0, Get(states, r, c, pitchState));
    }
}

__global__ void dropoutKernel(
    const float* A, float* B, float* C, curandStatePhilox4_32_10_t* states, const float dropoutRate,
    const std::size_t pitchA, const std::size_t pitchB, const std::size_t pitchC, const std::size_t pitchState,
    const std::size_t row, const std::size_t col) {
    
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    if(r < row && c < col) {
        curandStatePhilox4_32_10_t localState = *Get(states, r, c, pitchState);
        
        float randVal = curand_uniform(&localState);

        float mask = (randVal > dropoutRate) ? __frcp_rn(1.0f - dropoutRate) : 0.0f;

        *Get(C, r, c, pitchC) = mask * *Get(A, r, c, pitchA);
        *Get(B, r, c, pitchB) = mask;
    }
}