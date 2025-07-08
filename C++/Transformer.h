#ifndef TRANSFORMER
#define TRANSFORMER

#include "Header.h"
#include "Tensor.h"
#include "Encoder.h"
#include "Softmax.h"
#include "PositionalEncoder.h"
#include "Embedding.h"
#include "Linear.h"
#include "LayerNorm.h"
#include "Util.h"
#include "Decoder.h"

class Transformer {
public:
	Transformer();

    void forward(
        const int inpute[batch * sequenceLength], const int inputd[batch * sequenceLength], TensorView output,
        const int srcSeq[batch], const int tgtSeq[batch]);

	void predict(
        const int inpute[batch * sequenceLength], const int inputd[batch * sequenceLength], TensorView output,
        const int srcSeq[batch], const int tgtSeq[batch]);

	void backward(
        TensorView outputGradient,
        const int inpute[batch * sequenceLength], const int inputd[batch * sequenceLength],
        const int srcSeq[batch], const int tgtSeq[batch]);

	void updateParameter();

    void loadParam(cnpy::npz_t npFile, std::string prefix);

	void checkUpdatedParam(cnpy::npz_t npFile, std::string prefix);

	void forwardTest(cnpy::npz_t npFile, std::string prefix);

	void backwardTest(cnpy::npz_t npFile, std::string prefix);

    Embedding srcEmbed;
    Embedding tgtEmbed;
    PositionalEncoder srcPos;
    PositionalEncoder tgtPos;
    Decoder decoder;
    Encoder encoder;
    Linear linear;

	Tensor encoderOut;
    Tensor encoderGradient;
    Tensor out1;
    Tensor out2;
    Tensor out3;
    Tensor out4;
    Tensor out5;

    Tensor gradient1;
    Tensor gradient2;
    Tensor gradient3;
    Tensor gradient4;
    Tensor gradient5;
};

#endif
