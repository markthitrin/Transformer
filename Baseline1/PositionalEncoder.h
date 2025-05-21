#ifndef POSITIONAL_ENCODER
#define POSITIONAL_ENCODER

#include "Header.h"

class PositionalEncoder {
public:
	PositionalEncoder();

	Tensor operator()(const int sequenceLength, const Tensor& tensor) noexcept;
	Tensor predict(const int sequenceLength, const Tensor& tensor) noexcept;
	Tensor backpropagate(const Tensor& gradient) const noexcept;
};

#endif
