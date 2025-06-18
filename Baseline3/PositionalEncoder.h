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

	DropOut<batch * len, col, dropoutRate> dropout;
	
	Tensor<batch * len, col> _input;
	Tensor<batch * len, col>& _output;
	Tensor<batch * len, col>& _inGradient;
	Tensor<batch * len, col>& _outGradient;

	Tensor<batch * len, col> _positionEncode;

	Tensor<batch * len, col> _out1;
};

#endif
