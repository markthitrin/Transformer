#ifndef RELU
#define RELU

#include "Header.h"

class ReLU {
public:
	Tensor operator()(const Tensor& tensor) noexcept;
	Tensor predict(const Tensor& tensor) noexcept;
	Tensor backpropagate(const Tensor& gradient) const noexcept;

	Tensor x = Tensor(0, 0, 0);
};

#endif // !RELU
