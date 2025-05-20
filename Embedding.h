#ifndef EMBEDDING
#define EMBEDDING

#include "Header.h"

class Embedding {
public:
	Embedding();
	Embedding(const int numtokens, const int outshape) noexcept;

	Tensor operator()(const Tensor& tensor) noexcept;
	Tensor predict(const Tensor& tensor) noexcept;
	Tensor backpropagate(const Tensor& gradient) noexcept;
	void updateParameter() noexcept;

	int numToken						= 0;
	int outshape						= 0;
	std::vector<Matrix> table			= std::vector<Matrix>();

	int feedCount = 0;
	int t = 1;
	std::vector<Matrix> tableGradient	= std::vector<Matrix>();
	std::vector<Matrix> tableM			= std::vector<Matrix>();
	std::vector<Matrix> tableV			= std::vector<Matrix>();
	std::vector<bool> used				= std::vector<bool>();

	Tensor x							= Tensor(0, 0, 0);
};

#endif
