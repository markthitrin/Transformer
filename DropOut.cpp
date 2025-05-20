#include "Header.h"

DropOut::DropOut() { ; }

DropOut::DropOut(const float dropoutRate) noexcept : dropoutRate(dropoutRate) { ; }

Tensor DropOut::operator()(const Tensor& tensor) noexcept {
	Tensor result(tensor.batch, tensor.data[0].row, 1);
	mask = Tensor(tensor.batch, tensor.data[0].row, 1);
	for (int i = 0; i < tensor.batch; i++) {
		int threshold = dropoutRate * (1 << 10);
		for (int j = 0; j < tensor.data[0].row;j++) {
			mask[i][j][0] = ((std::rand() & ((1 << 11) - 1)) < threshold);
		}
		for (int j = 0; j < tensor.data[0].row; j++) {
			result[i][j][0] = mask[i][j][0] * tensor[i][j][0] / (1.0f - dropoutRate);
		}
	}
	return result;
}

Tensor DropOut::predict(const Tensor& tensor) noexcept {
	Tensor result(tensor.batch, tensor.data[0].row, 1);
	for (int i = 0; i < tensor.batch; i++) {
		for (int j = 0; j < tensor.data[0].row; j++) {
			result[i][j][0] = dropoutRate * tensor[i][j][0] / (1.0f - dropoutRate);
		}
	}
	return result;
}


Tensor DropOut::backpropagate(const Tensor& gradient) noexcept {
	Tensor result(gradient.batch, gradient.data[0].row, 1);
	for (int i = 0; i < gradient.batch; i++) {
		for (int j = 0; j < gradient.data[0].row; j++) {
			result[i][j][0] = gradient[i][j][0] * mask[i][j][0] / (1.0f - dropoutRate);
		}
	}
	return result;
}
