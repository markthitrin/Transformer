#include "Header.h"

Tensor ReLU::operator()(const Tensor& tensor) noexcept {
	x = tensor;
	Tensor result(tensor.batch, tensor.data[0].row, 1);
	for (int i = 0; i < tensor.batch; i++) {
		for (int j = 0; j < tensor.data[0].row; j++) {
			result[i][j][0] = tensor[i][j][0] * (tensor[i][j][0] >= 0);
		}
	}
	return result;
}

Tensor ReLU::predict(const Tensor& tensor) noexcept {
	x = tensor;
	Tensor result(tensor.batch, tensor.data[0].row, 1);
	for (int i = 0; i < tensor.batch; i++) {
		for (int j = 0; j < tensor.data[0].row; j++) {
			result[i][j][0] = tensor[i][j][0] * (tensor[i][j][0] >= 0);
		}
	}
	return result;
}


Tensor ReLU::backpropagate(const Tensor& gradient) const noexcept {
	Tensor result(gradient.batch, gradient.data[0].row, 1);
	for (int i = 0; i < gradient.batch; i++) {
		for (int j = 0; j < gradient.data[0].row; j++) {
			result[i][j][0] = gradient[i][j][0] * (x[i][j][0] >= 0);
		}
	}
	return result;
}