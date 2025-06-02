#ifndef DECODER_LAYER
#define DECODER_LAYER

#include "Header.h"
#include "Tensor.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "MultiheadAttention.h"
#include "Dropout.h"
#include "PositionwiseFeedForward.h"
#include "Util.h"

template<int d,int l,int col>
class DecoderLayer {
public:
	DecoderLayer() noexcept :
		_input		(norm1._input),
		_output		(dropout2._output),
		_inGradient (dropout2._inGradient),
		_outGradient(norm1._outGradient),
		_out1		(Create0<d * l, col>()),
		_out2		(Create0<d * l, col>()),
		_out3		(Create0<d * l, col>()),
		_out4		(Create0<d * l, col>()),
		_out5		(Create0<d * l, col>()),
		_gradient1	(Create0<d * l, col>()), 
		_gradient2	(Create0<d * l, col>()), 
		_gradient3	(Create0<d * l, col>()), 
		_gradient4	(Create0<d * l, col>()), 
		_gradient5	(Create0<d * l, col>()) {

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
	~DecoderLayer() {
		_aligned_free(_out1);
		_aligned_free(_out2);
		_aligned_free(_out3);
		_aligned_free(_out4);
		_aligned_free(_out5);
		_aligned_free(_gradient1);
		_aligned_free(_gradient2);
		_aligned_free(_gradient3);
		_aligned_free(_gradient4);
		_aligned_free(_gradient5);
	}

	void forward() noexcept {
		IMPORT_CONST(input);
		IMPORT(out3);
		IMPORT(output);
		norm1.forward();
		mulAtt.forward();
		dropout1.forward();
		Plus<d* l, col>((Tensor)input, (Tensor)out3, (Tensor)out3);
		norm2.forward();
		pff.forward();
		dropout2.forward();
		Plus<d* l, col>((Tensor)out3, (Tensor)output, (Tensor)output);
	}

	void predict() noexcept {
		IMPORT_CONST(input);
		IMPORT(out3);
		IMPORT(output);
		norm1.predict();
		mulAtt.predict();
		dropout1.predict();
		Plus<d* l, col>((Tensor)input, (Tensor)out3, (Tensor)out3);
		norm2.predict();
		pff.predict();
		dropout2.predict();
		Plus<d* l, col>((Tensor)out3, (Tensor)output, (Tensor)output);
	}

	void backpropagate() noexcept {
		IMPORT_CONST(inGradient);
		IMPORT(outGradient);
		IMPORT(gradient3);
		dropout2.backpropagate();
		pff.backpropagate();
		norm2.backpropagate();
		Plus<d, col>((Tensor)inGradient, (Tensor)gradient3, (Tensor)gradient3);
		dropout1.backpropagate();
		mulAtt.backpropagate();
		norm1.backpropagate();
		Plus<d*l, col>((Tensor)gradient3, (Tensor)outGradient, (Tensor)outGradient);
	}

	void updateParameter() noexcept {
		norm1.updateParameter();
		mulAtt.updateParameter();
		norm2.updateParameter();
		pff.updateParameter();
	}

	LayerNorm<d * l, col>							norm1;
	MultiheadAttention<head, d, l, col, col, col>	mulAtt;
	DropOut<d* l, col, 0.1>							dropout1;
	LayerNorm<d* l, col>							norm2;
	PositionwiseFeedForward<d* l, col, dFF, col>	pff;
	DropOut<d* l, col, 0.1>							dropout2;

	Tensor& _input;
	Tensor& _output;
	Tensor& _inGradient;
	Tensor& _outGradient;

	Tensor _out1;
	Tensor _out2;
	Tensor _out3;
	Tensor _out4;
	Tensor _out5;

	Tensor _gradient1;
	Tensor _gradient2;
	Tensor _gradient3;
	Tensor _gradient4;
	Tensor _gradient5;
};

#endif // !ENCODER_LAYER