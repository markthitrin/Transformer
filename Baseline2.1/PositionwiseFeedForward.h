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

	void loadParam(cnpy::npz_t npFile, std::string prefix) {
		linear1._weight.loadNp(npFile, prefix + ".w1");
		linear1._bias.loadNp(npFile, prefix + ".b1");
		linear2._weight.loadNp(npFile, prefix + ".w2");
		linear2._bias.loadNp(npFile, prefix + ".b2");
	}

	void checkUpdatedParam(cnpy::npz_t npFile, std::string prefix) {
		Tensor<hid, col> w1Updated;
		Tensor<1, hid> b1Updated;
		Tensor<col, hid> w2Updated;
		Tensor<1, col> b2Updated;
		w1Updated.init();
		b1Updated.init();
		w2Updated.init();
		b2Updated.init();
		w1Updated.loadNp(npFile, prefix + ".updated_w1");
		b1Updated.loadNp(npFile, prefix + ".updated_b1");
		w2Updated.loadNp(npFile, prefix + ".updated_w2");
		b2Updated.loadNp(npFile, prefix + ".updated_b2");

		PrintTestResult("backward " + prefix + ".w1", linear1._weight, w1Updated);
		PrintTestResult("backward " + prefix + ".b1", linear1._bias, b1Updated);
		PrintTestResult("backward " + prefix + ".w2", linear2._weight, w2Updated);
		PrintTestResult("backward " + prefix + ".b2", linear2._bias, b2Updated);
	}

	void forwardTest(cnpy::npz_t npFile, std::string prefix) {
		_input.init();
		_output.init();
		Tensor<row, col> target;
		target.init();

		_input.loadNp(npFile, prefix + ".input");
		target.loadNp(npFile, prefix + ".output");

		forward();
		PrintTestResult("forward",_output, target);
	}

	void backwardTest(cnpy::npz_t npFile, std::string prefix) {
		_inGradient.init();
		Set(_inGradient,1.0f / row / col);
		_outGradient.init();
		_input.init();
		_input.loadNp(npFile, prefix + ".input");
		_output.init();
		
		forward();
		backward();
		updateParameter();

		checkUpdatedParam(npFile, prefix);
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