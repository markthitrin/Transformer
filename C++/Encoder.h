#ifndef ENCODER
#define ENCODER

#include "Header.h"
#include "Tensor.h"
#include "EncoderLayer.h"
#include "Softmax.h"
#include "Embedding.h"
#include "Linear.h"
#include "Util.h"

class Encoder {
public:
	Encoder();

	void forward(TensorView input, TensorView output, const int srcSeq[batch]);

	void predict(TensorView input, TensorView output, const int srcSeq[batch]);

	void backward(TensorView outputGradient, TensorView inputGradient, const int srcSeq[batch]);

	void updateParameterTask();

    EncoderLayer layers[N];
    LayerNorm norm;

    std::vector<Tensor> outi;
	
    std::vector<Tensor> gradient;
};

#endif
