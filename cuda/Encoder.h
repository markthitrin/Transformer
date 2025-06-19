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
	Encoder(
		Tensor<batch * len, col>& input,
		Tensor<batch * len, col>& output,
		Tensor<batch * len, col>& outputGradient,
		Tensor<batch * len, col>& inputGradient) noexcept:
		norm(_out[N - 1], output, outputGradient, _gradient[N - 1]),
		_input(input),
		_output(output),
		_outputGradient(outputGradient),
		_inputGradient(inputGradient) {
		
		layers[0] = new EncoderLayer<batch, len, col>(input, _out[0], _gradient[0], _inputGradient);
		for(int i = 0;i < N - 1;i++) {
            layers[i + 1] = new EncoderLayer<batch, len, col>(_out[i], _out[i + 1], _gradient[i + 1], _gradient[i]);
        }
	}
	~Encoder() {
		for(int i = 0;i < N;i++) {
			delete layers[i];
			_out[i].free();
            _gradient[i].free();
		}
	}

	void forward(int npd) noexcept {
		for(int i = 0;i < N;i++) {
            layers[i]->forward(npd);
        }
        norm.forward();
	}

	void predict(int npd) noexcept {
		for(int i = 0;i < N;i++) {
            layers[i]->predict(npd);
        }
        norm.predict();
	}

	void backward(int npd) noexcept {
		norm.backward();
        for(int i = N - 1;i>=0;i--) {
            layers[i]->backward(npd);
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
		Tensor<batch * len, col> target;
		Tensor<1,1> npdLoad;

		_input.loadNp(npFile, prefix + ".input");
		target.loadNp(npFile, prefix + ".output");
		npdLoad.loadNp(npFile, prefix + ".npd");
		forward(npdLoad.data[0]);

		PrintTestResult("forward", _output, target);
	}

	void backwardTest(cnpy::npz_t npFile, std::string prefix) {
		Set(_outputGradient,1.0f / batch / len / col);

		_input.loadNp(npFile, prefix + ".input");
		
		Tensor<1, 1> npdLoader;
		npdLoader.loadNp(npFile, prefix + ".npd");

		forward(npdLoader.data[0]);
		backward(npdLoader.data[0]);
		updateParameter();

		checkUpdatedParam(npFile, prefix);
	}

	EncoderLayer<batch, len, col>* layers[N];
	LayerNorm<batch * len, col> norm;

	Tensor<batch * len, col>& _input;
	Tensor<batch * len, col>& _output;
	Tensor<batch * len, col>& _outputGradient;
    Tensor<batch * len, col>& _inputGradient;

	Tensor<batch * len, col> _out[N];
	
	Tensor<batch * len, col> _gradient[N];
};

#endif // !CLONE
