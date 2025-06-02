#ifndef RELU
#define RELU

#include "Header.h"
#include "Tensor.h"

template<int d,int col>
class ReLU {
public:
	ReLU() noexcept { ; }

	void forward() noexcept {
		IMPORT_CONST(input);
		IMPORT(output);
		for (int i = 0; i < d * col; i++) {
			output[i] = input[i] * float(input[i] >= 0);
		}
	}

	void predict() noexcept {
		forward();
	}

	void backpropagate() const noexcept {
		IMPORT_CONST(inGradient);
		IMPORT_CONST(input);
		IMPORT(outGradient);
		for (int i = 0; i < d * col; i++) {
			outGradient[i] = inGradient[i] * float(input[i] >= 0);
		}
	}

	Tensor _inGradient;
	Tensor _outGradient;
	Tensor _input;
	Tensor _output;
};

#endif // ! LOG_SOFTMAX
