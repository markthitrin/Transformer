#ifndef POSITIONAL_ENCODER
#define POSITIONAL_ENCODER

#include "Header.h"
#include "Tensor.h"
#include "Util.h"
#include "DropOut.h"

class PositionalEncoder {
public:
	PositionalEncoder();

	void forward(TensorView input, TensorView output);

    void predict(TensorView input, TensorView output);

    void backward(TensorView outputGradient, TensorView inputGradient);

    DropOut dropout;

	Tensor positionEncode;
};

#endif
