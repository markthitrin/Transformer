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
	PositionwiseFeedForward(
		Tensor<row, col>& input,
		Tensor<row, col>& output,
		Tensor<row, col>& inGradient,
		Tensor<row, col>& outGradient) noexcept:
		linear1(input, _out1, _gradient1, outGradient),
		relu(_out1, _out2, _gradient2, _gradient1),
		dropout(_out2, _out3, _gradient3, _gradient2),
		linear2(_out3, output, inGradient, _gradient3),
		_input(input),
		_output(output),
		_inGradient(inGradient),
		_outGradient(outGradient) { ; }
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
		Tensor<row, col> target;

		_input.loadNp(npFile, prefix + ".input");
		target.loadNp(npFile, prefix + ".output");

		forward();
		PrintTestResult("forward",_output, target);
	}

	void backwardTest(cnpy::npz_t npFile, std::string prefix) {
		Set(_inGradient,1.0f / row / col);

		_input.loadNp(npFile, prefix + ".input");
		
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