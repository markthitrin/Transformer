#ifndef ATTENTION
#define ATTENTION

#include "Header.h"

class Attention {	
public:
	Attention() noexcept;
	Attention(const int inshape, const int qshape, const int outshape) noexcept;

	Tensor operator()(const int sequenceLength, const Tensor& tensorQ, const Tensor& tensorK, const Tensor& tensorV, const Matrix& mask) noexcept;
	Tensor predict(const int sequenceLength, const Tensor& tensorQ, const Tensor& tensorK, const Tensor& tensorV, const Matrix& mask) noexcept;
	Tensor backpropagate(const int sequenceLength, const Tensor& gradient, const Matrix& mask) noexcept;
	void updateParameter() noexcept;

	int inshape = 0;
	int qshape = 0;
	int outshape = 0;

	Matrix WQ = Matrix(0, 0);
	Matrix WK = Matrix(0, 0);
	Matrix WV = Matrix(0, 0);

	int t = 1;
	int feedCount = 0;
	Matrix WQGradient = Matrix(0, 0);
	Matrix WQM = Matrix(0, 0);
	Matrix WQV = Matrix(0, 0);
	Matrix WKGradient = Matrix(0, 0);
	Matrix WKM = Matrix(0, 0);
	Matrix WKV = Matrix(0, 0);
	Matrix WVGradient = Matrix(0, 0);
	Matrix WVM = Matrix(0, 0);
	Matrix WVV = Matrix(0, 0);

	Tensor Q = Tensor(0, 0, 0);
	Tensor K = Tensor(0, 0, 0);
	Tensor V = Tensor(0, 0, 0);
	Tensor S = Tensor(0, 0, 0);
	Tensor A = Tensor(0, 0, 0);
	Tensor xQ = Tensor(0, 0, 0);
	Tensor xK = Tensor(0, 0, 0);
	Tensor xV = Tensor(0, 0, 0);
};

#endif