#ifndef SOFTMAX
#define SOFTMAX

#include "Header.h"
#include "Tensor.h"

class Softmax {
public:
	Softmax();

	void forward(TensorView input, TensorView output);

	void predict(TensorView input, TensorView output);

	void backward(TensorView outputGradient, TensorView inputGradient, TensorView output);
};

#endif