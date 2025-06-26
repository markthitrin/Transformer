#include "Header.cuh"
#include "Tensor.cuh"
#include "Util.cuh"
#include "UtilKernel.cuh"
#include "Softmax.cuh"

Softmax::Softmax(
    Tensor input,
    Tensor output,
    Tensor outputGradient,
    Tensor inputGradient) noexcept:

    input(input),
    output(output),
    outputGradient(outputGradient),
    inputGradient(inputGradient) { ; }

cudaGraphNode_t Softmax::AppendGraphForward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {

}

cudaGraphNode_t Softmax::AppendGraphPredict(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {

}

cudaGraphNode_t Softmax::AppendGraphBackward(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {

}

cudaGraphNode_t Softmax::AppendGraphUpdateParameter(cudaGraph_t graph, const std::vector<cudaGraphNode_t>& dependencyNodes) {

}

__global__ void sumExp(
    const float* A, const float* maxValue, float* C,
    const std::size_t pitchA, const std::size_t pitchC,
    const std::size_t row, const std::size_t col) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = threadIdx.x;

    __shared__ float buffer[REDUCTION_BLOCKSIZE_Y][REDUCTION_BLOCKSIZE_X];

	float acc = 0.0;
	for(std::size_t i = 0;i < ceil(col, REDUCTION_BLOCKSIZE_X);i++) {
		if(r < row && c + i * REDUCTION_BLOCKSIZE_X < col) {
			acc += expf(*Get(A, r, c + i * REDUCTION_BLOCKSIZE_X, pitchA) - *Get(maxValue, 0, r, 0));
		}
	}
	buffer[r][c] = acc;
	for(std::size_t i = REDUCTION_BLOCKSIZE_X / 2;i > 0;i /= 2) {
		__syncthreads();
		if(c < i) {
			buffer[r][c] += buffer[r][c + i];
		}
	}
	if(c == 0) {
		*Get(C, 0, r, 0) = buffer[r][0];
	}
}