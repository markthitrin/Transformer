#ifndef FEEDFORWARD_BLOCK
#define FEEDFORWARD_BLOCK

#include "Header.h"
#include "Tensor.h"
#include "Linear.h"
#include "ReLU.h"
#include "Util.h"
#include "LayerNorm.h"
#include "DropOut.h"

class FeedForwardBlock {
public:
	FeedForwardBlock();

	void forward(TensorView input, TensorView output);

	void predict(TensorView input, TensorView output);

	void backward(TensorView outputGradient, TensorView inputGradient, TensorView input);

	void updateParameter();

	void loadParam(cnpy::npz_t npFile, std::string prefix);

	void checkUpdatedParam(cnpy::npz_t npFile, std::string prefix);

	void forwardTest(cnpy::npz_t npFile, std::string prefix);

	void backwardTest(cnpy::npz_t npFile, std::string prefix);

	Linear linear1;
	ReLU relu;
	DropOut dropout;
	Linear linear2;

	Tensor out1;
	Tensor out2;
	Tensor out3;

	Tensor gradient1;
	Tensor gradient2;
	Tensor gradient3;
};

#endif