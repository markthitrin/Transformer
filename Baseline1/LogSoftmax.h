#ifndef LOG_SOFTMAX
#define LOG_SOFTMAX

#include "Header.h"

class LogSoftmax {
public:
	Tensor operator()(const Tensor& tensor) noexcept;
	Tensor predict(const Tensor& tensor) noexcept;
	Tensor backpropagate(const Tensor& gradient) const noexcept;

	Tensor y = Tensor(0, 0, 0);
};

#endif // ! LOG_SOFTMAX
