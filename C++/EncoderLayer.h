#ifndef ENCODER_LAYER
#define ENCODER_LAYER

#include "Header.h"
#include "Tensor.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "MultiheadAttention.h"
#include "DropOut.h"
#include "FeedForwardBlock.h"
#include "Util.h"

class EncoderLayer {
public:
	EncoderLayer();

	void forward(TensorView input, TensorView output, const int srcSeq[batch]);

	void predict(TensorView input, TensorView output, const int srcSeq[batch]);

	void backward(TensorView outputGradient, TensorView inputGradient, const int srcSeq[batch]);

	void updateParameterTask();

	LayerNorm norm1;
	MultiheadAttention mulAtt;
	DropOut dropout1;
	LayerNorm norm2;
	FeedForwardBlock pff;
	DropOut	dropout2;

	Tensor out1;
	Tensor out2;
	Tensor out3;
	Tensor out4;
	Tensor out5;

	Tensor gradient1;
	Tensor gradient2;
	Tensor gradient3;
	Tensor gradient4;
	Tensor gradient5;
};

#endif