#ifndef SOFTMAX
#define SOFTMAX

#include "Header.h"

class Softmax {
public:
	Tensor operator()(const Tensor& tensor) noexcept;
	Tensor predict(const Tensor& tensor) noexcept;
	Tensor backpropagate(const Tensor& gradient) const noexcept;

	Tensor y;
};

#endif