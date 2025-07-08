#ifndef MULTIHEAD_ATTENTION
#define MULTIHEAD_ATTENTION

#include "Header.h"
#include "Tensor.h"
#include "Util.h"
#include "Softmax.h"
#include "DropOut.h"

enum MaskType {
	LOOK_AHEAD,
	PADDING,
	CROSS_PADDING
};

class MultiheadAttention {
public:
	MultiheadAttention();

    void process(
        TensorView inputQ, TensorView inputK, TensorView inputV, TensorView output,
        MaskType maskType, const int seq[batch], bool train);

	void forward(
        TensorView inputQ, TensorView inputK, TensorView inputV, TensorView output,
        MaskType maskType, const int seq[batch]);

	void predict(
        TensorView inputQ, TensorView inputK, TensorView inputV, TensorView output,
        MaskType maskType, const int seq[batch]);

	void backward(
        TensorView outputGradient, TensorView inputGradientQ, TensorView inputGradientK, TensorView inputGradientV,
        TensorView inputQ, TensorView inputK, TensorView inputV, TensorView output,
        MaskType maskType, const int seq[batch]);

	void updateParameter();

    Softmax softmax;
	DropOut dropout;

	Tensor WQ;
	Tensor WK;
	Tensor WV;
	Tensor WO;

	AdamOptimizer WQOpt;
	AdamOptimizer WKOpt;
	AdamOptimizer WVOpt;
	AdamOptimizer WOOpt;

	Tensor QT;
	Tensor KT;
	Tensor VT;
	Tensor A;
	Tensor As;
    Tensor Ad;
	Tensor OT;

	Tensor QTGradient;
	Tensor KTGradient;
	Tensor VTGradient;
	Tensor AGradient;
	Tensor AsGradient;
    Tensor AdGradient;
	Tensor OTGradient;
};

#endif
