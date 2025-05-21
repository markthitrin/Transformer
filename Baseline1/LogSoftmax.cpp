#include "Header.h"

Tensor LogSoftmax::operator()(const Tensor& tensor) noexcept {
	Tensor result(tensor.batch, tensor.data[0].row, 1);
	for (int i = 0; i < tensor.batch; i++) {
		float maxValue = -FLT_MAX;
		for (int j = 0; j < tensor.data[0].row; j++) {
			maxValue = std::max(maxValue, tensor[i][j][0]);
		}

		float sumExp = 0;
		for (int j = 0; j < tensor.data[0].row; j++) {
			sumExp += std::exp(tensor[i][j][0] - maxValue);
		}
		float logSumExp = maxValue + std::log(sumExp);
		for (int j = 0; j < tensor.data[0].row; j++) {
			result[i][j][0] = tensor[i][j][0] - logSumExp;
		}
	}
	return y = result;
}

Tensor LogSoftmax::predict(const Tensor& tensor) noexcept {
	Tensor result(tensor.batch, tensor.data[0].row, 1);
	for (int i = 0; i < tensor.batch; i++) {
		float maxValue = -FLT_MAX;
		for (int j = 0; j < tensor.data[0].row; j++) {
			maxValue = std::max(maxValue, tensor[i][j][0]);
		}

		float sumExp = 0;
		for (int j = 0; j < tensor.data[0].row; j++) {
			sumExp += std::exp(tensor[i][j][0] - maxValue);
		}
		float logSumExp = maxValue + std::log(sumExp);
		for (int j = 0; j < tensor.data[0].row; j++) {
			result[i][j][0] = tensor[i][j][0] - logSumExp;
		}
	}
	return y = result;
}


Tensor LogSoftmax::backpropagate(const Tensor& gradient) const noexcept {
	Tensor result(gradient.batch, gradient.data[0].row, 1);
	Tensor expY(gradient.batch, gradient.data[0].row, 1);
	for (int i = 0; i < gradient.batch; i++) {
		float dot = 0;
		for (int j = 0; j < gradient.data[0].row; j++) {
			expY[i][j][0] = std::exp(y[i][j][0]);
		}
		for (int j = 0; j < gradient.data[0].row; j++) {
			dot += gradient[i][j][0] * expY[i][j][0];
		}
		for (int j = 0; j < gradient.data[0].row; j++) {
			result[i][j][0] = gradient[i][j][0] - dot * expY[i][j][0];
		}
	}
	return result;
}
