#ifndef DECODER_LAYER
#define DECODER_LAYER

#include "Header.h"
#include "Tensor.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "MultiheadAttention.h"
#include "DropOut.h"
#include "FeedForwardBlock.h"
#include "Util.h"

class DecoderLayer {
public:
	DecoderLayer();

	void forward(
         TensorView input, TensorView encoderOutput, TensorView output, 
        const int srcSeq[batch], const int tgtSeq[batch]);

	void predict(
         TensorView input, TensorView encoderOutput, TensorView output,  
        const int srcSeq[batch], const int tgtSeq[batch]);

	void backward(
        TensorView outputGradient, TensorView encoderGradient, TensorView inputGradient,
        TensorView encoderOutput, const int srcSeq[batch], const int tgtSeq[batch]);

    void updateParameterTask();

	LayerNorm norm1;
	MultiheadAttention mulAtt1;
	DropOut dropout1;

	LayerNorm norm2;
    MultiheadAttention mulAtt2;
    DropOut dropout2;

    LayerNorm norm3;
	FeedForwardBlock pff;
	DropOut	dropout3;

	Tensor out1;
	Tensor out2;
	Tensor out3;
	Tensor out4;
	Tensor out5;
	Tensor out6;
	Tensor out7;
	Tensor out8;

	Tensor gradient1;
	Tensor gradient2;
	Tensor gradient3;
	Tensor gradient4;
	Tensor gradient5;
	Tensor gradient6;
	Tensor gradient7;
	Tensor gradient8;
};

#endif