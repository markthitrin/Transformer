#ifndef LOG_SOFTMAX
#define LOG_SOFTMAX

#include "Header.h"
#include "Tensor.h"
#include "Util.h"

template<int d,int col>
class LogSoftmax {
public:
	LogSoftmax() noexcept { ; }

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
			float logSumExp = maxValue + fast_logf(sumExp);
			for (int j = 0; j < col; j++) {
				output[i * col + j] = input[i * col + j] - logSumExp;
			}
		}
	}

	void predict() noexcept {
		forward();
	}

	void backpropagate() const noexcept {
		IMPORT_CONST(inGradient);
		IMPORT_CONST(output);
		IMPORT(outGradient);
		for (int i = 0; i < d; i++) {
			float sum = 0.0f;
			for (int j = 0; j < col; j++) {
				outGradient[i * col + j] = std::exp(output[i * col + j]);
				sum += inGradient[i * col + j] * outGradient[i * col + j];
			}
			for (int j = 0; j < col; j++) {
				outGradient[i * col + j] = inGradient[i * col + j] - outGradient[i * col + j] * sum;
			}
		}
	}

	Tensor _inGradient;
	Tensor _outGradient;
	Tensor _input;
	Tensor _output;
};

#endif // ! LOG_SOFTMAX
