#ifndef LINEAR
#define LINEAR

#include "Header.h"
#include "Tensor.h"
#include "Util.h"

class Linear {
public:
	Linear(const int inD, const int outD);

    void forward(TensorView input, TensorView output);

    void predict(TensorView input, TensorView output);

    void backward(TensorView outputGradient, TensorView inputGradient, TensorView input);

    void updateParameterTask();

	Tensor weight;
	Tensor bias;

	AdamOptimizer weightOpt;
	AdamOptimizer biasOpt;
};

#endif
