#ifndef RELU
#define RELU

#include "Header.h"
#include "Tensor.h"

class ReLU {
public:
	ReLU() { ; }

	void forward(TensorView input, TensorView output);

	void predict(TensorView input, TensorView output);

	void backward(TensorView outputGradient, TensorView inputGradient);

    Tensor mask;
};

#endif
