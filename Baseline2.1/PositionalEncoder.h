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
		Tensor<batch * len, col>& inGradient,
		Tensor<batch * len, col>& outGradient) noexcept:
		dropout(output, output, inGradient, outGradient),
		_input(input),
		_output(output),
		_inGradient(inGradient),
		_outGradient(outGradient) {

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
	Tensor<batch * len, col>& _inGradient;
	Tensor<batch * len, col>& _outGradient;

	Tensor<batch * len, col> _positionEncode;
};

#endif
