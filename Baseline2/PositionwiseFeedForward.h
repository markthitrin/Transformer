#ifndef POSITIONWISE_FEED_FORWARD
#define POSITIONWISE_FEED_FORWARD

#include "Header.h"
#include "Tensor.h"
#include "Linear.h"
#include "ReLU.h"
#include "Util.h"
#include "LayerNorm.h"
#include "DropOut.h"

template<int d,int in,int hid,int out>
class PositionwiseFeedForward {
public:
	PositionwiseFeedForward() noexcept :
		_input		(linear1._input),
		_output		(linear2._output),
		_inGradient (linear2._inGradient),
		_outGradient(linear1._outGradient),
		_out1		(Create<d, hid>()),
		_out2		(Create<d, hid>()),
		_out3		(Create<d, hid>()),
		_gradient1	(Create0<d, hid>()), 
		_gradient2	(Create0<d, hid>()), 
		_gradient3	(Create0<d, hid>()) {

		relu._input = linear1._output = _out1;
		linear1._inGradient = relu._outGradient = _gradient1;

		dropout._input = relu._output = _out2;
		relu._inGradient = dropout._outGradient = _gradient2;

		linear2._input = dropout._output = _out3;
		dropout._inGradient = linear2._outGradient = _gradient3;
	}
	~PositionwiseFeedForward() {
		_aligned_free(_out1);
		_aligned_free(_out2);
		_aligned_free(_out3);
		_aligned_free(_gradient1);
		_aligned_free(_gradient2);
		_aligned_free(_gradient3);
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

	void backpropagate() noexcept {
		linear2.backpropagate();
		dropout.backpropagate();
		relu.backpropagate();
		linear1.backpropagate();
	}

	void updateParameter() noexcept {
		linear1.updateParameter();
		linear2.updateParameter();
	}

	Linear<d,in,hid> linear1;
	ReLU<d,hid> relu;
	DropOut<d, hid, 0.1f> dropout;
	Linear<d, hid, in> linear2;

	Tensor& _input;
	Tensor& _output;
	Tensor& _inGradient;
	Tensor& _outGradient;

	Tensor _out1;
	Tensor _out2;
	Tensor _out3;

	Tensor _gradient1;
	Tensor _gradient2;
	Tensor _gradient3;
};

#endif