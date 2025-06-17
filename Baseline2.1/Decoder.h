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

	void forward(int npdSrc, int npdTgt) noexcept {
		for(int i = 0;i < N;i++) {
            layers[i].forward(npdSrc, npdTgt);
        }
        norm.forward();
	}

	void predict(int npdSrc, int npdTgt) noexcept {
		for(int i = 0;i < N;i++) {
            layers[i].predict(npdSrc, npdTgt);
        }
        norm.predict();
	}

	void backward(int npdSrc, int npdTgt) noexcept {
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

    void loadParam(cnpy::npz_t npFile, std::string prefix) {
		for(int i = 0;i < N;i++) {
            layers[i].loadParam(npFile, prefix + ".layer" + std::to_string(i));
        }
        norm.loadParam(npFile, prefix + ".norm");
	}

	void checkUpdatedParam(cnpy::npz_t npFile, std::string prefix) {
		for(int i = 0;i < N;i++) {
            layers[i].checkUpdatedParam(npFile, prefix + ".layer" + std::to_string(i));
        }
        norm.checkUpdatedParam(npFile, prefix + ".norm");
	}

	void forwardTest(cnpy::npz_t npFile, std::string prefix) {
		_input.init();
        Tensor<batch * len, col> input2;
        Tensor<batch * len, col> outGradient;
        input2.init();
        outGradient.init();
        setEncodePtrTo(input2, outGradient);

		_output.init();
		Tensor<batch * len, col> target;
		Tensor<batch * len, col> targetsub;
		Tensor<1,2> npdLoad;
		target.init();
        targetsub.init();
		npdLoad.init();

		_input.loadNp(npFile, prefix + ".input1");
        input2.loadNp(npFile, prefix + ".input2");
		target.loadNp(npFile, prefix + ".output");
        targetsub.loadNp(npFile, prefix + ".outputsub");
		npdLoad.loadNp(npFile, prefix + ".npd");
		forward(npdLoad.data[0], npdLoad.data[1]);

		PrintTestResult("forward", _output, target);
		PrintTestResult("forward", _out[N - 1], targetsub);
	}

	void backwardTest(cnpy::npz_t npFile, std::string prefix) {
		_inGradient.init();
		Set(_inGradient,1.0f / batch / len / col);
		_outGradient.init();
		_input.init();
        Tensor<batch * len, col> input2;
        Tensor<batch * len, col> outGradient;
        input2.init();
        outGradient.init();
        setEncodePtrTo(input2, outGradient);

		_output.init();
		Tensor<1,2> npdLoad;
		npdLoad.init();

		_input.loadNp(npFile, prefix + ".input1");
        input2.loadNp(npFile, prefix + ".input2");
		npdLoad.loadNp(npFile, prefix + ".npd");

		forward(npdLoad.data[0], npdLoad.data[1]);
		backward(npdLoad.data[0], npdLoad.data[1]);
		updateParameter();

		checkUpdatedParam(npFile, prefix);
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
