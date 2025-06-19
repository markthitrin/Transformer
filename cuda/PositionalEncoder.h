#ifndef POSITIONAL_ENCODER
#define POSITIONAL_ENCODER

#include "Header.h"
#include "Tensor.h"
#include "Util.h"
#include "cnpy.h"
#include "DropOut.h"

template<int batch,int len,int col>
class PositionalEncoder {
public:
	PositionalEncoder(
		Tensor<batch * len, col>& input,
		Tensor<batch * len, col>& output,
		Tensor<batch * len, col>& outputGradient,
		Tensor<batch * len, col>& inputGradient) noexcept:
		dropout(output, output, outputGradient, inputGradient),
		_input(input),
		_output(output),
		_outputGradient(outputGradient),
		_inputGradient(inputGradient) {

		GetPositionalEncode<batch, len, col>(_positionEncode);
	}

	void forward() noexcept {
		Plus(_input, _positionEncode, _output);
		dropout.forward();
	}

	void predict() noexcept {
		Plus(_input, _positionEncode, _output);
		dropout.predict();
	}

	void backward() noexcept {
		dropout.backward();
	}

	void forwardTest(cnpy::npz_t npFile, std::string prefix) {
		float error = 0.0f;

		Tensor<batch * len, col> target;

		_input.loadNp(npFile, prefix + ".input");
		target.loadNp(npFile, prefix + ".output");

		forward();

		PrintTestResult("forward " + prefix, _output, target);
	}

	DropOut<batch * len, col, dropoutRate> dropout;
	
	Tensor<batch * len, col>& _input;
	Tensor<batch * len, col>& _output;
	Tensor<batch * len, col>& _outputGradient;
	Tensor<batch * len, col>& _inputGradient;

	Tensor<batch * len, col> _positionEncode;
};

#endif
