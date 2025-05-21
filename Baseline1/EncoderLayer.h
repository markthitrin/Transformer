#ifndef ENCODER_LAYER
#define ENCODER_LAYER

#include "Header.h"

class EncoderLayer {
public:
	EncoderLayer() noexcept;
	EncoderLayer(const int dModel) noexcept;

	Tensor operator()(const int sequenceLength, const Tensor& tensor, const Matrix& mask) noexcept;
	Tensor predict(const int sequenceLength, const Tensor& tensor, const Matrix& mask) noexcept;
	Tensor backpropagate(const int sequenceLength, const Tensor& gradient, const Matrix& mask) noexcept;
	void updateParameter() noexcept;

	LayerNorm norm1 = LayerNorm();
	MultiheadAttention mulAtt = MultiheadAttention();
	DropOut dropOut1 = DropOut();
	LayerNorm norm2 = LayerNorm();
	PositionwiseFeedForward pff = PositionwiseFeedForward();
	DropOut dropOut2 = DropOut();
};

#endif // !ENCODER_LAYER