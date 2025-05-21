#ifndef MULTIHEAD_ATTENTION
#define MULTIHEAD_ATTENTION

#include "Header.h"

class MultiheadAttention {
public:
	MultiheadAttention() noexcept;
	MultiheadAttention(const int head, const int dModel, const int qshape) noexcept;

	Tensor operator()(const int sequenceLength, const Tensor& tensorQ, const Tensor& tensorK, const Tensor& tensorV, const Matrix& mask) noexcept;
	Tensor predict(const int sequenceLength, const Tensor& tensorQ, const Tensor& tensorK, const Tensor& tensorV, const Matrix& mask) noexcept;
	Tensor backpropagate(const int sequenceLength, const Tensor& gradient, const Matrix& mask) noexcept;
	void updateParameter() noexcept;

	std::vector<Attention> attLayer = std::vector<Attention>();

	int head			= 0;
	int sequenceLength	= 0;
	int dModel			= 0;
	int qshape			= 0;
	int vshape			= 0;

	Matrix WO			= Matrix(0, 0);

	Matrix WOGradient	= Matrix(0, 0);
	Matrix WOM			= Matrix(0, 0);
	Matrix WOV			= Matrix(0, 0);

	int feedCount		= 0;
	int t				= 1;
	Tensor Attout		= Tensor(0, 0, 0);
};

#endif // !MULTIHEAD_ATTENTION
