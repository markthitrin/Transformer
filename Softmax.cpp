#include "Header.h"

Tensor Softmax::operator()(const Tensor& tensor) noexcept {
	Tensor result(tensor.batch, tensor.data[0].row, 1);
	for (int i = 0; i < tensor.batch; i++) {
		float sumExp = 0.0;
		float maxValue = -FLT_MAX;
		for (int j = 0; j < tensor.data[0].row; j++) {
			maxValue = std::max(tensor[i][j][0], maxValue);
		}
		for (int j = 0; j < tensor.data[0].row; j++) {
			sumExp += std::exp(tensor[i][j][0] - maxValue);
		}
		for (int j = 0; j < tensor.data[0].row; j++) {
			result[i][j][0] = std::exp(tensor[i][j][0] - maxValue) / sumExp;
		}
	}
	return y = result;
}

Tensor Softmax::predict(const Tensor& tensor) noexcept {
	Tensor result(tensor.batch, tensor.data[0].row, 1);
	for (int i = 0; i < tensor.batch; i++) {
		float sumExp = 0.0;
		float maxValue = -FLT_MAX;
		for (int j = 0; j < tensor.data[0].row; j++) {
			maxValue = std::max(tensor[i][j][0], maxValue);
		}
		for (int j = 0; j < tensor.data[0].row; j++) {
			sumExp += std::exp(tensor[i][j][0] - maxValue);
		}
		for (int j = 0; j < tensor.data[0].row; j++) {
			result[i][j][0] = std::exp(tensor[i][j][0] - maxValue) / sumExp;
		}
	}
	return y = result;
}


Tensor Softmax::backpropagate(const Tensor& gradient) const noexcept {
	Tensor result(gradient.batch, gradient.data[0].row, 1);
	for (int i = 0; i < gradient.batch; i++) {
		for (int j = 0; j < gradient.data[0].row; j++) {
			for (int k = 0; k < gradient.data[0].row; k++) {
				result[i][j][0] += y[i][k][0] * (float(j == k) - y[i][j][0]) * gradient[i][k][0];
			}
		}
	}
	return result;
}
