#ifndef ENCODER
#define ENCODER

#include "Header.h"
#include "Tensor.h"
#include "EncoderLayer.h"
#include "Softmax.h"
#include "PositionalEncoder.h"
#include "Embedding.h"
#include "Linear.h"
#include "LayerNorm.h"
#include "Util.h"

template<int batch,int len,int col,int N>
class Encoder {
public:
	Encoder() noexcept :
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
	~Encoder() {
		for(int i = 0;i < N;i++) {
            _out[i].free();
            _gradient[i].free();
        }
	}

	void forward(int npd) noexcept {
		for(int i = 0;i < N;i++) {
            layers[i].forward(npd);
        }
        norm.forward();
	}

	void predict(int npd) noexcept {
		for(int i = 0;i < N;i++) {
            layers[i].predict(npd);
        }
        norm.predict();
	}

	void backward(int npd) noexcept {
		norm.backward();
        for(int i = N - 1;i>=0;i--) {
            layers[i].backward(npd);
        }
	}

	void updateParameter() noexcept {
        for(int i = 0;i < N;i++) {
            layers[i].updateParameter();
        }
		norm.updateParameter();
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
		_output.init();
		Tensor<batch * len, col> target;
		Tensor<1,1> npdLoad;
		target.init();
		npdLoad.init();

		_input.loadNp(npFile, prefix + ".input");
		target.loadNp(npFile, prefix + ".output");
		npdLoad.loadNp(npFile, prefix + ".npd");
		forward(npdLoad.data[0]);

		PrintTestResult("forward", _output, target);
	}

	void backwardTest(cnpy::npz_t npFile, std::string prefix) {
		_inGradient.init();
		Set(_inGradient,1.0f / batch / len / col);
		_outGradient.init();
		_input.init();
		_input.loadNp(npFile, prefix + ".input");
		_output.init();
		
		Tensor<1, 1> npdLoader;
		npdLoader.init();
		npdLoader.loadNp(npFile, prefix + ".npd");

		forward(npdLoader.data[0]);
		backward(npdLoader.data[0]);
		updateParameter();

		checkUpdatedParam(npFile, prefix);
	}

    EncoderLayer<batch, len, col> layers[N];
	LayerNorm<batch * len, col> norm;

	Tensor<batch * len, col>& _input;
	Tensor<batch * len, col>& _output;
	Tensor<batch * len, col>& _inGradient;
    Tensor<batch * len, col>& _outGradient;

	Tensor<batch * len, col> _out[N];
	
	Tensor<batch * len, col> _gradient[N];
};

#endif // !CLONE
