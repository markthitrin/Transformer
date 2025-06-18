#ifndef RELU
#define RELU

#include "Header.h"
#include "Tensor.h"

template<int row, int col>
class ReLU {
public:
	ReLU(
		Tensor<row, col>& input,
		Tensor<row, col>& output,
		Tensor<row, col>& inGradient,
		Tensor<row, col>& outGradient) noexcept:
		_input(input),
		_output(output),
		_inGradient(inGradient),
		_outGradient(outGradient)  { ; }

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

	void backward() noexcept {
		IMPORT_CONST(inGradient);
		IMPORT_CONST(input);
		IMPORT(outGradient);

		for (int i = 0; i < row * col; i++) {
			outGradient[i] = inGradient[i] * float(input[i] >= 0);
		}
	}

	Tensor<row, col>& _input;
	Tensor<row, col>& _output;
	Tensor<row, col>& _inGradient;
	Tensor<row, col>& _outGradient;
};

#endif // ! LOG_SOFTMAX
