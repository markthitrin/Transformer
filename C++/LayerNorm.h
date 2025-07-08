#ifndef LAYER_NORM
#define LAYER_NORM

#include "Header.h"
#include "Tensor.h"
#include "Util.h"

class LayerNorm {
public:
	LayerNorm();

	void forward(TensorView input, TensorView output);

	void predict(TensorView input, TensorView output);

	void backward(TensorView outputGradient, TensorView inputGradient);

	void updateParameter();

	Tensor alpha;
	Tensor bias;

	AdamOptimizer alphaOpt;
	AdamOptimizer biasOpt;

	Tensor xHat;
	Tensor std;
};

#endif