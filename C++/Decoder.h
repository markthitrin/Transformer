#ifndef DECODER
#define DECODER

#include "Header.h"
#include "Tensor.h"
#include "DecoderLayer.h"
#include "Softmax.h"
#include "Embedding.h"
#include "Linear.h"
#include "Util.h"

class Decoder {
public:
	Decoder();

	void forward(
		TensorView input, TensorView encoderOutput, TensorView output,
		const int srcSeq[batch], const int tgtSeq[batch]);

	void predict(
		TensorView input, TensorView encoderOutput, TensorView output,
		const int srcSeq[batch], const int tgtSeq[batch]);

	void backward(
        TensorView outputGradient, TensorView inputGradient, TensorView encoderGradient, TensorView encoderOutput,
        const int srcSeq[batch], const int tgtSeq[batch]);

	void updateParameterTask();

    void loadParam(cnpy::npz_t npFile, std::string prefix);

	void checkUpdatedParam(cnpy::npz_t npFile, std::string prefix);

	void forwardTest(cnpy::npz_t npFile, std::string prefix);

	void backwardTest(cnpy::npz_t npFile, std::string prefix);

    DecoderLayer layers[N];
    LayerNorm norm;

    std::vector<Tensor> outi;
	
    std::vector<Tensor> gradient;
};

#endif
