#include "Header.h"

PositionalEncoder::PositionalEncoder() { ; }

Tensor PositionalEncoder::operator()(const int sequenceLength, const Tensor& tensor) noexcept {
	int batch = tensor.batch / sequenceLength;
	Tensor result(tensor.batch, tensor.data[0].row, 1);
	for (int i = 0; i < batch; i++) {
		int s = sequenceLength * i;
		for (int j = 0; j < sequenceLength; j++) {
			for (int k = 0; k < tensor.data[0].row; k+=2) {
				result[s + j][k][0] = std::sin(j / std::pow(10000, float(k) / tensor.data[0].row));
			}
			for (int k = 1; k < tensor.data[0].row; k += 2) {
				result[s + j][k][0] = std::cos(j / std::pow(10000, float(k - 1) / tensor.data[0].row));
			}
		}
	}
	return tensor + result;
}

Tensor PositionalEncoder::predict(const int sequenceLength, const Tensor& tensor) noexcept {
	int batch = tensor.batch / sequenceLength;
	Tensor result(tensor.batch, tensor.data[0].row, 1);
	for (int i = 0; i < batch; i++) {
		int s = sequenceLength * i;
		for (int j = 0; j < sequenceLength; j++) {
			for (int k = 0; k < tensor.data[0].row; k += 2) {
				result[s + j][k][0] = std::sin(j / std::pow(10000, float(k) / tensor.data[0].row));
			}
			for (int k = 1; k < tensor.data[0].row; k += 2) {
				result[s + j][k][0] = std::cos(j / std::pow(10000, float(k - 1) / tensor.data[0].row));
			}
		}
	}
	return tensor + result;
}


Tensor PositionalEncoder::backpropagate(const Tensor& gradient) const noexcept {
	return gradient;
}