#ifndef DECODER_LAYER
#define DECODER_LAYER

#include "Header.h"
#include "Tensor.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "MultiheadAttention.h"
#include "DropOut.h"
#include "PositionwiseFeedForward.h"
#include "Util.h"

template<int batch,int len,int col>
class DecoderLayer {
public:
	DecoderLayer(
		Tensor<batch * len, col>& input,
		Tensor<batch * len, col>& encoderOut,
		Tensor<batch * len, col>& output,
		Tensor<batch * len, col>& outputGradient,
		Tensor<batch * len, col>& inputGradient,
		Tensor<batch * len, col>& encoderGradient) noexcept:
		norm1(input, _out1, _gradient1, inputGradient),
		mulAtt1(_out1, _out1, _out1, _out2, _gradient2, _gradient1, _gradient1, _gradient1),
		dropout1(_out2, _out3, _gradient3, _gradient2),
		norm2(_out3, _out4, _gradient4, _gradient3),
		mulAtt2(_out4, encoderOut, encoderOut, _out5, _gradient5, _gradient4, encoderGradient, encoderGradient),
		dropout2(_out5, _out6, _gradient6, _gradient5),
		norm3(_out6, _out7, _gradient7, _gradient6),
		pff(_out7, _out8, _gradient8, _gradient7),
		dropout3(_out8, output, outputGradient, _gradient8),
		_input(input),
		_encoderOut(encoderOut),
		_output(output),
		_outputGradient(outputGradient),
		_inputGradient(inputGradient),
		_encoderGradient(encoderGradient) {;}
	~DecoderLayer() {
        _out1.free();
        _out2.free();
        _out3.free();
        _out4.free();
        _out5.free();
        _out6.free();
        _out7.free();
        _out8.free();
        _gradient1.free();
        _gradient2.free();
        _gradient3.free();
        _gradient4.free();
        _gradient5.free();
        _gradient6.free();
        _gradient7.free();
        _gradient8.free();
    }

	void forward(int npdSrc, int npdTgt) noexcept {
		norm1.forward();
		mulAtt1.forward(npdTgt);
		dropout1.forward();
		Plus(_input, _out3, _out3);

		norm2.forward();
		mulAtt2.forward(npdSrc);
		dropout2.forward();
		Plus(_out3, _out6, _out6);

        norm3.forward();
		pff.forward();
		dropout3.forward();
		Plus(_out6, _output, _output);
	}

	void predict(int npdSrc, int npdTgt) noexcept {
		norm1.predict();
		mulAtt1.predict(npdTgt);
		dropout1.predict();
		Plus(_input, _out3, _out3);

		norm2.predict();
		mulAtt2.predict(npdSrc);
		dropout2.predict();
		Plus(_out3, _out6, _out6);

        norm3.predict();
		pff.predict();
		dropout3.predict();
		Plus(_out6, _output, _output);
	}

	void backward(int npdSrc, int npdTgt) noexcept {
		dropout3.backward();
		pff.backward();
		norm3.backward();
		Plus(_outputGradient, _gradient6, _gradient6);

        dropout2.backward();
		mulAtt2.backward(npdSrc);
		norm2.backward();
		Plus(_gradient6, _gradient3, _gradient3);

		dropout1.backward();
		mulAtt1.backward(npdTgt);
		norm1.backward();
		Plus(_gradient3, _inputGradient, _inputGradient);
	}

	void updateParameter() noexcept {
		norm1.updateParameter();
		mulAtt1.updateParameter();
		norm2.updateParameter();
        mulAtt2.updateParameter();
        norm3.updateParameter();
		pff.updateParameter();
	}

	void loadParam(cnpy::npz_t npFile, std::string prefix) {
		norm1.loadParam(npFile, prefix + ".sub1.layerNorm");
		mulAtt1.loadParam(npFile, prefix + ".sub1.sublayer");
        norm2.loadParam(npFile, prefix + ".sub2.layerNorm");
        mulAtt2.loadParam(npFile, prefix + ".sub2.sublayer");
		norm3.loadParam(npFile, prefix + ".sub3.layerNorm");
		pff.loadParam(npFile,prefix + ".sub3.sublayer");
	}

	void checkUpdatedParam(cnpy::npz_t npFile, std::string prefix) {
		norm1.checkUpdatedParam(npFile, prefix + ".sub1.layerNorm");
		mulAtt1.checkUpdatedParam(npFile, prefix + ".sub1.sublayer");
		norm2.checkUpdatedParam(npFile, prefix + ".sub2.layerNorm");
		mulAtt2.checkUpdatedParam(npFile, prefix + ".sub2.sublayer");
        norm3.checkUpdatedParam(npFile, prefix + ".sub3.layerNorm");
        pff.checkUpdatedParam(npFile, prefix + ".sub3.sublayer");
	}

	void forwardTest(cnpy::npz_t npFile, std::string prefix) {
		Tensor<batch * len, col> target;
		Tensor<batch * len, col> target1;
		Tensor<batch * len, col> target2;
		Tensor<1,2> npdLoad;

		_input.loadNp(npFile, prefix + ".input1");
        mulAtt2._inputK.loadNp(npFile, prefix + ".input2");
		target.loadNp(npFile, prefix + ".output");
		target1.loadNp(npFile, prefix + ".output1");
		target2.loadNp(npFile, prefix + ".output2");
		npdLoad.loadNp(npFile, prefix + ".npd");
		forward(npdLoad.data[0], npdLoad.data[1]);

		PrintTestResult("forward", _output, target);
		PrintTestResult("forward1", _out3, target1);
		PrintTestResult("forward2", _out6, target2);
	}

	void backwardTest(cnpy::npz_t npFile, std::string prefix) {
		Set(_outputGradient,1.0f / batch / len / col);
        mulAtt2._inputV = mulAtt2._inputK;
        mulAtt2._inputGradientV = mulAtt2._inputGradientK;
		Tensor<batch * len, col> target;
		Tensor<1,2> npdLoad;

		_input.loadNp(npFile, prefix + ".input1");
        mulAtt2._inputK.loadNp(npFile, prefix + ".input2");
		target.loadNp(npFile, prefix + ".output");
		npdLoad.loadNp(npFile, prefix + ".npd");
		forward(npdLoad.data[0], npdLoad.data[1]);
		backward(npdLoad.data[0], npdLoad.data[1]);
		updateParameter();

		checkUpdatedParam(npFile, prefix);
	}

	LayerNorm<batch * len, col> norm1;
	MultiheadAttention<head, 0, batch, len, col> mulAtt1;
	DropOut<batch * len, col, dropoutRate> dropout1;

	LayerNorm<batch * len, col> norm2;
    MultiheadAttention<head, 2, batch, len, col> mulAtt2;
    DropOut<batch * len, col, dropoutRate> dropout2;

    LayerNorm<batch * len, col> norm3;
	PositionwiseFeedForward<batch * len, col, dFF> pff;
	DropOut<batch * len, col, dropoutRate>	dropout3;

	Tensor<batch * len, col>& _input;
	Tensor<batch * len, col>& _encoderOut;
	Tensor<batch * len, col>& _output;
	Tensor<batch * len, col>& _outputGradient;
	Tensor<batch * len, col>& _inputGradient;
	Tensor<batch * len, col>& _encoderGradient;

	Tensor<batch * len, col> _out1;
	Tensor<batch * len, col> _out2;
	Tensor<batch * len, col> _out3;
	Tensor<batch * len, col> _out4;
	Tensor<batch * len, col> _out5;
	Tensor<batch * len, col> _out6;
	Tensor<batch * len, col> _out7;
	Tensor<batch * len, col> _out8;

	Tensor<batch * len, col> _gradient1;
	Tensor<batch * len, col> _gradient2;
	Tensor<batch * len, col> _gradient3;
	Tensor<batch * len, col> _gradient4;
	Tensor<batch * len, col> _gradient5;
	Tensor<batch * len, col> _gradient6;
	Tensor<batch * len, col> _gradient7;
	Tensor<batch * len, col> _gradient8;
};

#endif // !ENCODER_LAYER