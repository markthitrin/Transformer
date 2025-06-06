#ifndef SOFTMAX
#define SOFTMAX

#include "Header.h"
#include "Tensor.h"

template<int d,int col>
class Softmax {
public:
	Softmax() noexcept { ; }

	void forward() noexcept {
		IMPORT_CONST(input);
		IMPORT(output);
		for (int i = 0; i < d; i++) {
			float sumExp = 0.0;
			float maxValue = -FLT_MAX;
			for (int j = 0; j < col; j++) {
				maxValue = std::max(maxValue, input[i * col + j]);
			}
			for (int j = 0; j < col; j++) {
				sumExp += std::exp(input[i * col + j] - maxValue);
			}
			for (int j = 0; j < col; j++) {
				output[i * col + j] = std::exp(input[i * col + j] - maxValue) / sumExp;
			}
		}
	}

	void predict() noexcept {
		forward();
	}

	void backpropagate() noexcept {
		IMPORT_CONST(inGradient);
		IMPORT_CONST(output);
		IMPORT(outGradient);
		for (int i = 0; i < d; i++) {
			float sumGY = 0.0f;
			for (int j = 0; j < col; j++) {
				sumGY += inGradient[i * col + j] * output[i * col + j];
			}
			for (int j = 0; j < col; j++) {
				outGradient[i * col + j] = output[i * col + j] * (inGradient[i * col + j] - sumGY);
			}
		}
	}
	
	Tensor _inGradient;
	Tensor _outGradient;
	Tensor _input;
	Tensor _output;
};

#endif