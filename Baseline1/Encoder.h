#ifndef ENCODER
#define ENCODER

#include "Header.h"

class Encoder {
public:
	Encoder();
	Encoder(const int dModel, const int N) noexcept;

	Tensor operator()(const int sequenceLength, const Tensor& tensor) noexcept;
	Tensor predict(const int sequenceLength, const Tensor& tensor) noexcept;
	Tensor backpropagate(const int sequenceLength, const Tensor& gradient) noexcept;
	void updateParameter() noexcept;

	std::vector<EncoderLayer> layers	= std::vector<EncoderLayer>();
	LayerNorm norm						= LayerNorm();
};

#endif // !CLONE
