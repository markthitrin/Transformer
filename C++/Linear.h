#ifndef LINEAR
#define LINEAR

#include "Header.h"
#include "Tensor.h"
#include "Util.h"

class Linear {
public:
	Linear(const int in, const int out);

    void forward(TensorView input, TensorView output);

    void predict(TensorView input, TensorView output);

    void backward(TensorView outputGradient, TensorView inputGradient, TensorView input);

    void updateParameter();

	void checkUpdatedParam(cnpy::npz_t npFile, std::string prefix);

	Tensor weight;
	Tensor bias;

	AdamOptimizer weightOpt;
	AdamOptimizer biasOpt;
};

#endif
