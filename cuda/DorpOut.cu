#include "Header.cuh"
#include "Tensor.cuh"

class DropOut {
public:
    DropOut(
        Tensor input,
		Tensor output,
		Tensor outputGradient,
		Tensor inputGradient) noexcept:
        input(input),
        output(output),
        outputGradient(outputGradient),
        inputGradient(inputGradient),
        mask(input.row, output.col) {;}
    ~DropOut() {
        mask.free();
    }

    cudaGraphNode_t AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {

    }
    
	cudaGraphNode_t AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {

    }

	cudaGraphNode_t AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {

    }

	cudaGraphNode_t AppendGraphUpdateParameter(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {

    }

    void forward() noexcept;

    void predict() noexcept;

    void backward() noexcept;

    Tensor input;
    Tensor output;
    Tensor outputGradient;
    Tensor inputGradient;

    Tensor mask;
};



cudaGraphNode_t AppendDropoutMaskCreateNode(
    cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes,
    Tensor mask) {
    
}



__global__ void setup_states(curandStatePhilox4_32_10_t* states, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void apply_dropout(float* data, int size, float p, curandStatePhilox4_32_10_t* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Use curand_uniform for [0,1)
    float rand = curand_uniform(&states[idx]);
    data[idx] *= (rand < p) ? (1.0f / p) : 0.0f;  // Scale to keep expected value same
}