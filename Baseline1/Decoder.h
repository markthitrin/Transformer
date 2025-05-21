#ifndef DECODER
#define DECODER

#include "Header.h"

class Decoder {
public:
	Decoder() ;
	Decoder(const int dModel, const int N, const int vocab) noexcept;

	Tensor operator()(const int sequenceLength, const Tensor& tensor) noexcept;
	Tensor predict(const int sequenceLength, const Tensor& tensor) noexcept;
	Tensor backpropagate(const int sequenceLength, const Tensor& gradient) noexcept;
	void updateParameter() noexcept;

	Embedding embedding					= Embedding();
	PositionalEncoder positionalEncoder = PositionalEncoder();
	std::vector<DecoderLayer> layers	= std::vector<DecoderLayer>();
	LayerNorm norm						= LayerNorm();
	Linear linear						= Linear();
	Softmax softmax						= Softmax();
};

#endif // !CLONE
