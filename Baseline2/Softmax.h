#ifndef SOFTMAX
#define SOFTMAX

#include "Header.h"

class Softmax {
public:
	Tensor operator()(const int d, const int row, Tensor tensor) noexcept;
	Tensor predict(const int d, const int row, const Tensor& tensor) noexcept;
	Tensor backpropagate(const int d, const int row, const Tensor& gradient) const noexcept;

	Tensor y;
};

#endif