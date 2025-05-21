#ifndef DROP_OUT
#define DROP_OUT

#include "Header.h"

class DropOut {
public:
	DropOut();
	DropOut(const float dropoutRate) noexcept;

	Tensor operator()(const Tensor& tensor) noexcept;
	Tensor predict(const Tensor& tensor) noexcept;
	Tensor backpropagate(const Tensor& gradient) noexcept;

	const float dropoutRate = 0.1;

	Tensor mask = Tensor(0, 0, 0);
};

#endif // !DROP_OUT
