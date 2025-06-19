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
	Decoder(
        Tensor<batch * len, col>& input,
        Tensor<batch * len, col>& encoderOut,
		Tensor<batch * len, col>& output,
		Tensor<batch * len, col>& inGradient,
		Tensor<batch * len, col>& outGradient,
        Tensor<batch * len, col>& encoderGradient) noexcept:
		norm(_out[N - 1], output, inGradient, _gradient[N - 1]),
		_input(input),
		_encoderOut(encoderOut),
		_output(output),
		_inGradient(inGradient),
		_outGradient(outGradient),
		_encoderGradient(encoderGradient)  {
		

        layers[0] = new DecoderLayer<batch, len, col>(input, encoderOut, _out[0], _gradient[0], _outGradient, encoderGradient);
		for(int i = 0;i < N - 1;i++) {
            layers[i + 1] = new DecoderLayer<batch, len, col>(_out[i], encoderOut, _out[i + 1], _gradient[i + 1], _gradient[i], encoderGradient);
        }
	}
	~Decoder() {
		for(int i  =0;i < N;i++) {
            _out[i].free();
            _gradient[i].free();
			delete layers[i];
		}
	}

	void forward(int npdSrc, int npdTgt) noexcept {
		for(int i = 0;i < N;i++) {
            layers[i]->forward(npdSrc, npdTgt);
        }
        norm.forward();
	}

	void predict(int npdSrc, int npdTgt) noexcept {
		for(int i = 0;i < N;i++) {
            layers[i]->predict(npdSrc, npdTgt);
        }
        norm.predict();
	}

	void backward(int npdSrc, int npdTgt) noexcept {
		norm.backward();
        for(int i = N - 1;i>=0;i--) {
            layers[i]->backward(npdSrc, npdTgt);
        }
	}

	void updateParameter() noexcept {
        for(int i = 0;i < N;i++) {
            layers[i]->updateParameter();
        }
		norm.updateParameter();
	}

    void loadParam(cnpy::npz_t npFile, std::string prefix) {
		for(int i = 0;i < N;i++) {
            layers[i]->loadParam(npFile, prefix + ".layer" + std::to_string(i));
        }
        norm.loadParam(npFile, prefix + ".norm");
	}

	void checkUpdatedParam(cnpy::npz_t npFile, std::string prefix) {
		for(int i = 0;i < N;i++) {
            layers[i]->checkUpdatedParam(npFile, prefix + ".layer" + std::to_string(i));
        }
        norm.checkUpdatedParam(npFile, prefix + ".norm");
	}

	void forwardTest(cnpy::npz_t npFile, std::string prefix) {
        Tensor<batch * len, col> outGradient;

		Tensor<batch * len, col> target;
		Tensor<batch * len, col> targetsub;
		Tensor<1,2> npdLoad;

		_input.loadNp(npFile, prefix + ".input1");
        _encoderOut.loadNp(npFile, prefix + ".input2");
		target.loadNp(npFile, prefix + ".output");
        targetsub.loadNp(npFile, prefix + ".outputsub");
		npdLoad.loadNp(npFile, prefix + ".npd");
		forward(npdLoad.data[0], npdLoad.data[1]);

		PrintTestResult("forward", _output, target);
		PrintTestResult("forwards", _out[N - 1], targetsub);
	}

	void backwardTest(cnpy::npz_t npFile, std::string prefix) {
		Set(_inGradient,1.0f / batch / len / col);
        Tensor<batch * len, col> outGradient;

		Tensor<1,2> npdLoad;

		_input.loadNp(npFile, prefix + ".input1");
        _encoderOut.loadNp(npFile, prefix + ".input2");
		npdLoad.loadNp(npFile, prefix + ".npd");

		forward(npdLoad.data[0], npdLoad.data[1]);
		backward(npdLoad.data[0], npdLoad.data[1]);
		updateParameter();

		checkUpdatedParam(npFile, prefix);
	}

    DecoderLayer<batch, len, col>* layers[N];
	LayerNorm<batch * len, col> norm;

	Tensor<batch * len, col>& _input;
	Tensor<batch * len, col>& _encoderOut;
	Tensor<batch * len, col>& _output;
	Tensor<batch * len, col>& _inGradient;
	Tensor<batch * len, col>& _outGradient;
	Tensor<batch * len, col>& _encoderGradient;

	Tensor<batch * len, col> _out[N];
	
	Tensor<batch * len, col> _gradient[N];
};

#endif // !CLONE
