#ifndef SOFTMAX
#define SOFTMAX

#include "Header.h"
#include "Tensor.h"

template<int row,int col>
class Softmax {
public:
	Softmax() noexcept { ; }

	void forward() noexcept {
        IMPORT_CONST(input);
        IMPORT(output);

		for (int i = 0; i < row; i++) {

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

	void backward() noexcept {
		IMPORT_CONST(inGradient);
		IMPORT_CONST(output);
		IMPORT(outGradient);

		for (int i = 0; i < col; i++) {
			float sumGY = 0.0f;

			for (int j = 0; j < col; j++) {
				sumGY += inGradient[i * col + j] * output[i * col + j];
			}

			for (int j = 0; j < col; j++) {
				outGradient[i * col + j] = output[i * col + j] * (inGradient[i * col + j] - sumGY);
			}
		}
	}
	
	Tensor<row, col> _inGradient;
	Tensor<row, col> _outGradient;
	Tensor<row, col> _input;
	Tensor<row, col> _output;
};

#endif