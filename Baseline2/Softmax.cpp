#include "Header.h"

Tensor Softmax::operator()(const int d1, const int d2, Tensor tensor) noexcept {
	for (int i = 0; i < d1; i++) {
		for (int j = 0; j < d2; j++) {

		}
	}
}

Tensor Softmax::predict(const int d1, const int d2, const Tensor& tensor) noexcept {

}

Tensor Softmax::backpropagate(const int d1, const int d2, const Tensor& gradient) const noexcept {

}
