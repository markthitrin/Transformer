#ifndef LAYER_NORM
#define LAYER_NORM

#include "Header.h"

class LayerNorm {
public:
	LayerNorm() noexcept;
	LayerNorm(const int shape) noexcept;

	Tensor operator()(const Tensor& tensor) noexcept;
	Tensor predict(const Tensor& tensor) noexcept;
	Tensor backpropagate(const Tensor& gradient) noexcept;
	void updateParameter() noexcept;

	int shape			= 0;
	Matrix y			= Matrix(0, 0);
	Matrix b			= Matrix(0, 0);

	int t				= 1;
	int feedCount		= 0;
	Matrix yGradient	= Matrix(0, 0);
	Matrix yM			= Matrix(0, 0);
	Matrix yV			= Matrix(0, 0);
	Matrix bGradient	= Matrix(0, 0);
	Matrix bM			= Matrix(0, 0);
	Matrix bV			= Matrix(0, 0);

	Tensor xHat			= Tensor(0, 0, 0);
	Tensor o2			= Tensor(0, 0, 0);
};

#endif