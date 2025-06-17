#ifndef RELU
#define RELU

#include "Header.h"
#include "Tensor.h"

template<int row, int col>
class ReLU {
public:
	ReLU() noexcept { ; }

	void forward() noexcept {
		IMPORT_CONST(input);
		IMPORT(output);

		for (int i = 0; i < row * col; i++) {
			output[i] = input[i] * float(input[i] >= 0);
		}
	}

	void predict() noexcept {
		forward();
	}

	void backward() const noexcept {
		IMPORT_CONST(inGradient);
		IMPORT_CONST(input);
		IMPORT(outGradient);

		for (int i = 0; i < row * col; i++) {
			outGradient[i] = inGradient[i] * float(input[i] >= 0);
		}
	}

	Tensor<row, col> _inGradient;
	Tensor<row, col> _outGradient;
	Tensor<row, col> _input;
	Tensor<row, col> _output;
};

#endif // ! LOG_SOFTMAX
