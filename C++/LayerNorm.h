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

	void updateParameterTask();

	void loadParam(cnpy::npz_t npFile, std::string prefix);

    void forwardTest(cnpy::npz_t npFile, std::string prefix);

	void checkUpdatedParam(cnpy::npz_t npFile, std::string prefix);

	void backwardTest(cnpy::npz_t npFile, std::string prefix);

	Tensor alpha;
	Tensor bias;

	AdamOptimizer alphaOpt;
	AdamOptimizer biasOpt;

	Tensor xHat;
	Tensor std;
};

#endif