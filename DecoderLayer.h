#ifndef DECODER_LAYER
#define DECODER_LAYER

#include "Header.h"

class DecoderLayer {
public:
	DecoderLayer() noexcept;
	DecoderLayer(const int dModel) noexcept;

	Tensor operator()(const int sequenceLength, const Tensor& tensor) noexcept;
	Tensor predict(const int sequenceLength, const Tensor& tensor) noexcept;
	Tensor backpropagate(const int sequenceLength, const Tensor& gradient) noexcept;
	void updateParameter() noexcept;

	LayerNorm norm1 = LayerNorm();
	MultiheadAttention mulAtt = MultiheadAttention();
	DropOut dropOut1 = DropOut();
	LayerNorm norm2 = LayerNorm();
	PositionwiseFeedForward pff = PositionwiseFeedForward();
	DropOut dropOut2 = DropOut();
};

#endif // !ENCODER_LAYER