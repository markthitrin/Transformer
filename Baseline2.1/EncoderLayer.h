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
	EncoderLayer() :
		_input		(norm1._input),
		_output		(dropout2._output),
		_inGradient (dropout2._inGradient),
		_outGradient(norm1._outGradient) {

        _out1.init();
		_out2.init();
		_out3.init();
		_out4.init();
		_out5.init();
		_gradient1.init(); 
		_gradient2.init();
		_gradient3.init();
		_gradient4.init();
		_gradient5.init();

		mulAtt._inputQ = mulAtt._inputK = mulAtt._inputV = norm1._output = _out1;
		norm1._inGradient = mulAtt._outGradientQ = mulAtt._outGradientK = mulAtt._outGradientV = _gradient1;
		
		dropout1._input = mulAtt._output = _out2;
		mulAtt._inGradient = dropout1._outGradient = _gradient2;

		norm2._input = dropout1._output = _out3;
		dropout1._inGradient = norm2._outGradient = _gradient3;

		pff._input = norm2._output = _out4;
		norm2._inGradient = pff._outGradient = _gradient4;

		dropout2._input = pff._output = _out5;
		pff._inGradient = dropout2._outGradient = _gradient5;

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
		_input.init();
		_output.init();
		Tensor<batch * len, col> target;
		Tensor<1,1> npdLoad;
		target.init();
		npdLoad.init();

		_input.loadNp(npFile, prefix + ".input");
		target.loadNp(npFile, prefix + ".output");
		npdLoad.loadNp(npFile, prefix + ".npd");
		forward(npdLoad.data[0]);

		PrintTestResult("forward", _output, target);
	}

	void backwardTest(cnpy::npz_t npFile, std::string prefix) {
		_inGradient.init();
		Set(_inGradient,1.0f / batch / len / col);
		_outGradient.init();
		_input.init();
		_input.loadNp(npFile, prefix + ".input");
		_output.init();
		
		Tensor<1, 1> npdLoader;
		npdLoader.init();
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