#ifndef DROP_OUT
#define DROP_OUT

#include "Header.cuh"
#include "Tensor.cuh"

class DropOut {
public:
    DropOut(
        Tensor& input,
		Tensor& output,
		Tensor& outputGradient,
		Tensor& inputGradient,
        const std::size_t row,
        const std::size_t col) noexcept;
    ~DropOut();

    cudaGraphNode_t AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);
    
	cudaGraphNode_t AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

	cudaGraphNode_t AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes);

    Tensor& input;
    Tensor& output;
    Tensor& outputGradient;
    Tensor& inputGradient;

    Tensor mask;

    curandStatePhilox4_32_10_t* states;
    std::size_t pitchState;
};

cudaGraphNode_t AppendDropoutNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor input, Tensor mask, Tensor output, curandStatePhilox4_32_10_t* states, float dropoutRate,
    std::size_t pitchState);

cudaGraphNode_t AppendDropoutBackwardNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor outputGradient, Tensor mask, Tensor inputGradient);

__global__ void setupState(
    curandStatePhilox4_32_10_t* states, unsigned long long seed,
    const std::size_t pitchState,
    const std::size_t row, const std::size_t col);

__global__ void dropoutKernel(
    const float* A, float* B, float* C, curandStatePhilox4_32_10_t* states, const float dropoutRate,
    const std::size_t pitchA, const std::size_t pitchB, const std::size_t pitchC, const std::size_t pitchState,
    const std::size_t row, const std::size_t col);

#endif // !DROP_OUT
