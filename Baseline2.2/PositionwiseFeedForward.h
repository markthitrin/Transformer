#ifndef POSITIONWISE_FEED_FORWARD
#define POSITIONWISE_FEED_FORWARD

#include "Header.h"
#include "Tensor.h"
#include "Linear.h"
#include "ReLU.h"
#include "Util.h"
#include "LayerNorm.h"
#include "DropOut.h"

template<int row,int col,int hid>
class PositionwiseFeedForward {
public:
	PositionwiseFeedForward() noexcept :
		_input		(linear1._input),
		_output		(linear2._output),
		_inGradient (linear2._inGradient),
		_outGradient(linear1._outGradient) {
		
		_out1.init();
		_out2.init();
		_out3.init();
		_gradient1.init();
		_gradient2.init();
		_gradient3.init();

		relu._input = linear1._output = _out1;
		linear1._inGradient = relu._outGradient = _gradient1;

		dropout._input = relu._output = _out2;
		relu._inGradient = dropout._outGradient = _gradient2;

		linear2._input = dropout._output = _out3;
		dropout._inGradient = linear2._outGradient = _gradient3;
	}
	~PositionwiseFeedForward() noexcept {
		_out1.free();
		_out2.free();
		_out3.free();
		_gradient1.free();
		_gradient2.free();
		_gradient3.free();
	}

	void forward() noexcept {
		linear1.forward();
		relu.forward();
		dropout.forward();
		linear2.forward();
	}

	void predict() noexcept {
		linear1.predict();
		relu.predict();
		dropout.predict();
		linear2.predict();
	}

	void backward() noexcept {
		linear2.backward();
		dropout.backward();
		relu.backward();
		linear1.backward();
	}

	void updateParameter() noexcept {
		linear1.updateParameter();
		linear2.updateParameter();
	}

	Linear<row, col, hid> linear1;
	ReLU<row, hid> relu;
	DropOut<row, hid, 0.1f> dropout;
	Linear<row, hid, col> linear2;

	Tensor<row, col>& _input;
	Tensor<row, col>& _output;
	Tensor<row, col>& _inGradient;
	Tensor<row, col>& _outGradient;

	Tensor<row, hid> _out1;
	Tensor<row, hid> _out2;
	Tensor<row, hid> _out3;

	Tensor<row, hid> _gradient1;
	Tensor<row, hid> _gradient2;
	Tensor<row, hid> _gradient3;
};

#endif