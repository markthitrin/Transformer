#ifndef EMBEDDING
#define EMBEDDING

#include "Header.h"
#include "Tensor.h"
#include "Util.h"

class Embedding {
public:
	Embedding(const int numTokens);

	void forward(const int input[batch * sequenceLength], TensorView output);

	void predict(const int input[batch * sequenceLength], TensorView output);

	void backward(TensorView outputGradient, const int* input);

	void updateParameterTask();

    void loadParam(cnpy::npz_t npFile, std::string prefix);

    void forwardTest(cnpy::npz_t npFile, std::string prefix);

	void checkUpdatedParam(cnpy::npz_t npFile, std::string prefix);

	void backwardTest(cnpy::npz_t npFile, std::string prefix);

	Tensor table;

	std::vector<AdamOptimizer> tableOpt;
    std::vector<bool> needUpdate;
};

#endif
