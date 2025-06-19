#ifndef ENCODER_LAYER
#define ENCODER_LAYER

#include "Header.h"
#include "Tensor.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "MultiheadAttention.h"
#include "DropOut.h"
#include "PositionwiseFeedForward.h"
#include "Util.h"

template<int batch,int len,int col>
class EncoderLayer {
public:
	EncoderLayer(
		Tensor<batch * len, col>& input,
		Tensor<batch * len, col>& output,
		Tensor<batch * len, col>& inGradient,
		Tensor<batch * len, col>& outGradient) noexcept:
		norm1(input, _out1, _gradient1, outGradient),
		mulAtt(_out1, _out1, _out1, _out2, _gradient2, _gradient1, _gradient1, _gradient1),
		dropout1(_out2, _out3, _gradient3, _gradient2),
		norm2(_out3, _out4, _gradient4, _gradient3),
		pff(_out4, _out5, _gradient5, _gradient4),
		dropout2(_out5, output, inGradient, _gradient5),
		_input(input),
		_output(output),
		_inGradient(inGradient),
		_outGradient(outGradient) {;}
	~EncoderLayer() {
		_out1.free();
		_out2.free();
		_out3.free();
		_out4.free();
		_out5.free();
		_gradient1.free();
		_gradient2.free();
		_gradient3.free();
		_gradient4.free();
		_gradient5.free();
	}

	void forward(int npd) noexcept {
		norm1.forward();
		mulAtt.forward(npd);
		dropout1.forward();
		Plus(_input, _out3, _out3);

		norm2.forward();
		pff.forward();
		dropout2.forward();
		Plus(_out3, _output, _output);
	}

	void predict(int npd) noexcept {
		norm1.predict();
		mulAtt.predict(npd);
		dropout1.predict();
		Plus(_input, _out3, _out3);

		norm2.predict();
		pff.predict();
		dropout2.predict();
		Plus(_out3, _output, _output);
	}

	void backward(int npd) noexcept {
		dropout2.backward();
		pff.backward();
		norm2.backward();
		Plus(_inGradient, _gradient3, _gradient3);

		dropout1.backward();
		mulAtt.backward(npd);
		norm1.backward();
		Plus(_gradient3, _outGradient, _outGradient);
	}

	void updateParameter() noexcept {
		norm1.updateParameter();
		mulAtt.updateParameter();
		norm2.updateParameter();
		pff.updateParameter();
	}

	void loadParam(cnpy::npz_t npFile, std::string prefix) {
		norm1.loadParam(npFile, prefix + ".sub1.layerNorm");
		mulAtt.loadParam(npFile, prefix + ".sub1.sublayer");
		norm2.loadParam(npFile, prefix + ".sub2.layerNorm");
		pff.loadParam(npFile,prefix + ".sub2.sublayer");
	}

	void checkUpdatedParam(cnpy::npz_t npFile, std::string prefix) {
		norm1.checkUpdatedParam(npFile, prefix + ".sub1.layerNorm");
		mulAtt.checkUpdatedParam(npFile, prefix + ".sub1.sublayer");
		norm2.checkUpdatedParam(npFile, prefix + ".sub2.layerNorm");
		pff.checkUpdatedParam(npFile, prefix + ".sub2.sublayer");
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
		Set(_inGradient,1.0f / batch / len / col);
		_input.loadNp(npFile, prefix + ".input");
		
		Tensor<1, 1> npdLoader;
		npdLoader.loadNp(npFile, prefix + ".npd");

		forward(npdLoader.data[0]);
		backward(npdLoader.data[0]);
		updateParameter();

		checkUpdatedParam(npFile, prefix);
	}

	LayerNorm<batch * len, col> norm1;
	MultiheadAttention<head, 1, batch, len, col> mulAtt;
	DropOut<batch * len, col, dropoutRate> dropout1;
	LayerNorm<batch * len, col> norm2;
	PositionwiseFeedForward<batch * len, col, dFF> pff;
	DropOut<batch * len, col, dropoutRate>	dropout2;

	Tensor<batch * len, col>& _input;
	Tensor<batch * len, col>& _output;
	Tensor<batch * len, col>& _inGradient;
	Tensor<batch * len, col>& _outGradient;

	Tensor<batch * len, col> _out1;
	Tensor<batch * len, col> _out2;
	Tensor<batch * len, col> _out3;
	Tensor<batch * len, col> _out4;
	Tensor<batch * len, col> _out5;

	Tensor<batch * len, col> _gradient1;
	Tensor<batch * len, col> _gradient2;
	Tensor<batch * len, col> _gradient3;
	Tensor<batch * len, col> _gradient4;
	Tensor<batch * len, col> _gradient5;
};

#endif // !ENCODER_LAYER