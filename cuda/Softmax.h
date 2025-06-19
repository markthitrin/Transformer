#ifndef SOFTMAX
#define SOFTMAX

#include "Header.h"
#include "Tensor.h"

class Softmax {
public:
	Softmax(
		Tensor input,
		Tensor output,
		Tensor outputGradient,
		Tensor inputGradient) noexcept:
		input(input),
		output(output),
		outputGradient(outputGradient),
		inputGradient(inputGradient) { ; }

	void forward() noexcept {
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
		for (int i = 0; i < col; i++) {
			float sumGY = 0.0f;

			for (int j = 0; j < col; j++) {
				sumGY += outputGradient[i * col + j] * output[i * col + j];
			}

			for (int j = 0; j < col; j++) {
				inputGradient[i * col + j] = output[i * col + j] * (outputGradient[i * col + j] - sumGY);
			}
		}
	}

	Tensor& input;
	Tensor& output;
	Tensor& outputGradient;
	Tensor& inputGradient;
};

#endif