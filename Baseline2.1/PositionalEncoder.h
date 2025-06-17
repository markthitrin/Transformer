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
	PositionalEncoder() : 
		_inGradient(dropout._inGradient),
		_outGradient(dropout._outGradient), 
		_output(dropout._output) {

		_positionEncode.init();
		GetPositionalEncode<batch, len, col>(_positionEncode);

		_out1.init();

		dropout._input = _out1;
	}
	~PositionalEncoder() {
		_out1.free();
	}

	void forward() noexcept {
		Plus(_input, _positionEncode, _out1);
		dropout.forward();
	}

	void predict() noexcept {
		Plus(_input, _positionEncode, _out1);
		dropout.predict();
	}

	void backward() noexcept {
		dropout.backward();
	}

	float forwardTest(cnpy::npz_t npFile, std::string prefix) {
		float error = 0.0f;

		_input.init();
		_output.init();
		Tensor<batch * len, col> target;
		target.init();

		_input.loadNp(npFile, prefix + ".input");
		target.loadNp(npFile, prefix + ".output");

		forward();
		for(int i  = 0;i < batch * len * col;i++) {
			if(i < 10) {
				std::cout << target.data[i] << " :: " << _output.data[i] << std::endl;
			}
			error += std::abs(target.data[i] - _output.data[i]);
		}
		return error / batch / len / col;
	}

	DropOut<batch * len, col, dropoutRate> dropout;
	
	Tensor<batch * len, col> _input;
	Tensor<batch * len, col>& _output;
	Tensor<batch * len, col>& _inGradient;
	Tensor<batch * len, col>& _outGradient;

	Tensor<batch * len, col> _positionEncode;

	Tensor<batch * len, col> _out1;
};

#endif
