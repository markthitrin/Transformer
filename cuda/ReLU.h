#ifndef RELU
#define RELU

#include "Header.h"
#include "Tensor.cuh"

__global__ void ReLUKernel(
	const float* A, const float* C,
	const std::size_t pitchA, std::size_t pitchC,
	const std::size_t row, const std::size_t col);

class ReLU {
public:
	ReLU(
		Tensor input,
		Tensor output,
		Tensor outputGradient,
		Tensor inputGradient) noexcept:
		input(input),
		output(output),
		outputGradient(outputGradient),
		inputGradient(inputGradient)  { ; }

	void forward() noexcept {
		constexpr int BLOCKSIZE = 16;
    	dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    	dim3 gridDim(ceil(input.col, BLOCKSIZE), ceil(input.row, BLOCKSIZE));
		ReLUKernel<<<gridDim, blockDim>>>(input.data, output.data, input.pitch, output.pitch, input.row, input.col);
	}

	void predict() noexcept {
		forward();
	}

	void backward() noexcept {
		
	}

	Tensor& input;
	Tensor& output;
	Tensor& outputGradient;
	Tensor& inputGradient;
};

__global__ void ReLUKernel(
	const float* A, const float* C,
	const std::size_t pitchA, const std::size_t pitchC,
	const std::size_t row, const std::size_t col) {

	const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

	if(r < row && c < col) {
		*Get(C, r, c, pitchC) = fmaxf(*Get(A, r, c, pitchA), 0.0f); 
	}
}

__global__ void ReLUBackwardKernel(
	const float* A, const float* B, const float* C,
	const std::size_t pitchA, const std::size_t pitchB, const std::size_t pitchC,
	const std::size_t row, const std::size_t col) {
	
	const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

	if(r < row && c < col) {
		*Get(C, r, c, pitchC) = float(*Get(B, r, c, pitchB) > 0) * Get(A, r, c, pitchA); 
	}
}


#endif // ! LOG_SOFTMAX
