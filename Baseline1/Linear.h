#ifndef LINEAR
#define LINEAR

#include "Header.h"

class Linear {
public:
	Linear();
	Linear(const int inshape, const int outshape);
	
	Tensor operator()(const Tensor& tensor) noexcept;
	Tensor predict(const Tensor& tensor) noexcept;
	Tensor backpropagate(const Tensor& gradient) noexcept;
	void updateParameter() noexcept;

	int inshape				= 0;
	int outshape			= 0;
	Matrix weight			= Matrix(0, 0);
	Matrix bias				= Matrix(0, 0);

	int t					= 1;
	int feedCount			= 0;
	Matrix weightGradient	= Matrix(0, 0);
	Matrix weightM			= Matrix(0, 0);
	Matrix weightV			= Matrix(0, 0);
	Matrix biasGradient		= Matrix(0, 0);
	Matrix biasM			= Matrix(0, 0);
	Matrix biasV			= Matrix(0, 0);

	Tensor x				= Tensor(0, 0, 0);
};

#endif // !LINEAR
