#ifndef DECODER
#define DECODER

#include "Header.h"
#include "Tensor.h"
#include "DecoderLayer.h"
#include "Softmax.h"
#include "PositionalEncoder.h"
#include "Embedding.h"
#include "Linear.h"
#include "LayerNorm.h"
#include "Util.h"

template<int batch,int len,int col,int N>
class Decoder {
public:
	Decoder() noexcept :
        _input(layers[0]._input),
        _output(norm._output),
        _inGradient(norm._inGradient),
        _outGradient(layers[0]._outGradient) {

        for(int i = 0;i < N;i++) {
            _out[i].init();
            _gradient[i].init();
        }
		
		for(int i = 0;i < N;i++) {
            layers[i]._output = _out[i];
            layers[i]._inGradient = _gradient[i];
        }
        for(int i = 0;i < N - 1;i++) {
            layers[i + 1]._input = _out[i];
            layers[i + 1]._outGradient = _gradient[i];
        }

        norm._input = _out[N - 1];
        norm._outGradient = _gradient[N - 1];
	}
	~Decoder() {
		for(int i = 0;i < N;i++) {
            _out[i].free();
            _gradient[i].free();
        }
	}

	void forward(int npdSrc[batch], int npdTgt[batch]) noexcept {
		for(int i = 0;i < N;i++) {
            layers[i].forward(npdSrc, npdTgt);
        }
        norm.forward();
	}

	void predict(int npdSrc[batch], int npdTgt[batch]) noexcept {
		for(int i = 0;i < N;i++) {
            layers[i].predict(npdSrc, npdTgt);
        }
        norm.predict();
	}

	void backward(int npdSrc[batch], int npdTgt[batch]) noexcept {
		norm.backward();
        for(int i = N - 1;i>=0;i--) {
            layers[i].backward(npdSrc, npdTgt);
        }
	}

	void updateParameter() noexcept {
        for(int i = 0;i < N;i++) {
            layers[i].updateParameter();
        }
		norm.updateParameter();
	}

    void setEncodePtrTo(Tensor<batch * len, col> input, Tensor<batch * len, col> outGradient) {
        for(int i = 0;i < N;i++) {
            layers[i].setEncodePtrTo(input, outGradient);
        }
    }

    DecoderLayer<batch, len, col> layers[N];
	LayerNorm<batch * len, col> norm;

	Tensor<batch * len, col>& _input;
	Tensor<batch * len, col>& _output;
	Tensor<batch * len, col>& _inGradient;
    Tensor<batch * len, col>& _outGradient;

	Tensor<batch * len, col> _out[N];
	
	Tensor<batch * len, col> _gradient[N];
};

#endif // !CLONE
