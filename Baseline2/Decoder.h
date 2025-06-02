#ifndef DECODER
#define DECODER

#include "Header.h"
#include "Tensor.h"
#include "DecoderLayer.h"
#include "Softmax.h"
#include "PositionalEncoder.h"
#include "Embedding.h"
#include "Linear.h"
#include "LayerNorm.h"
#include "Util.h"

template<int d,int token,int l,int col,int vocab>
class Decoder {
public:
	Decoder() noexcept :
		_input(embedding._input),
		_output(softmax._output),
		_inGradient(softmax._inGradient),
		_out1		(Create0<d* l, col>()),
		_out2		(Create0<d* l, col>()),
		_out3_1		(Create0<d* l, col>()),
		_out3_2		(Create0<d* l, col>()),
		_out3_3		(Create0<d* l, col>()),
		_out3_4		(Create0<d* l, col>()),
		_out3_5		(Create0<d* l, col>()),
		_out3_6     (Create0<d* l, col>()),
		_out4		(Create0<d* l, col>()),
		_out5		(Create0<d* l, vocab>()),
		_gradient2	(Create0<d* l, col>()),
		_gradient3_1(Create0<d* l, col>()),
		_gradient3_2(Create0<d* l, col>()),
		_gradient3_3(Create0<d* l, col>()),
		_gradient3_4(Create0<d* l, col>()),
		_gradient3_5(Create0<d* l, col>()),
		_gradient3_6(Create0<d* l, col>()),
		_gradient4	(Create0<d* l, col>()),
		_gradient5	(Create0<d* l, vocab>()) {
		
		positionalEncoder._input = embedding._output = _out1;
		layers1._input = positionalEncoder._output = _out2;
		embedding._inGradient = layers1._outGradient = _gradient2;

		layers2._input = layers1._output = _out3_1;
		layers1._inGradient = layers2._outGradient = _gradient3_1;

		layers3._input = layers2._output = _out3_2;
		layers2._inGradient = layers3._outGradient = _gradient3_2;

		layers4._input = layers3._output = _out3_3;
		layers3._inGradient = layers4._outGradient = _gradient3_3;

		layers5._input = layers4._output = _out3_4;
		layers4._inGradient = layers5._outGradient = _gradient3_4;

		layers6._input = layers5._output = _out3_5;
		layers5._inGradient = layers6._outGradient = _gradient3_5;

		norm._input = layers6._output = _out3_6;
		layers6._inGradient = norm._outGradient = _gradient3_6;

		linear._input = norm._output = _out4;
		norm._inGradient = linear._outGradient = _gradient4;

		softmax._input = linear._output = _out5;
		linear._inGradient = softmax._outGradient = _gradient5;
	}
	~Decoder() {
		_aligned_free(_out1);
		_aligned_free(_out2);
		_aligned_free(_out3_1);
		_aligned_free(_out3_2);
		_aligned_free(_out3_3);
		_aligned_free(_out3_4);
		_aligned_free(_out3_5);
		_aligned_free(_out3_6);
		_aligned_free(_out4);
		_aligned_free(_out5);

		_aligned_free(_gradient2);
		_aligned_free(_gradient3_1);
		_aligned_free(_gradient3_2);
		_aligned_free(_gradient3_3);
		_aligned_free(_gradient3_4);
		_aligned_free(_gradient3_5);
		_aligned_free(_gradient3_6);
		_aligned_free(_gradient4);
		_aligned_free(_gradient5);
	}

	void forward() noexcept {
		embedding.forward();
		positionalEncoder.forward();
		layers1.forward();
		layers2.forward();
		layers3.forward();
		layers4.forward();
		layers5.forward();
		layers6.forward();
		norm.forward();
		linear.forward();
		softmax.forward();
	}

	void predict() noexcept {
		embedding.predict();
		positionalEncoder.predict();
		layers1.predict();
		layers2.predict();
		layers3.predict();
		layers4.predict();
		layers5.predict();
		layers6.predict();
		norm.predict();
		linear.predict();
		softmax.predict();
	}

	void backpropagate() noexcept {
		softmax.backpropagate();
		linear.backpropagate();
		norm.backpropagate();
		layers6.backpropagate();
		layers5.backpropagate();
		layers4.backpropagate();
		layers3.backpropagate();
		layers2.backpropagate();
		layers1.backpropagate();
		embedding.backpropagate();
	}

	void updateParameter() noexcept {
		embedding.updateParameter();
		layers1.updateParameter();
		layers2.updateParameter();
		layers3.updateParameter();
		layers4.updateParameter();
		layers5.updateParameter();
		layers6.updateParameter();
		norm.updateParameter();
		linear.updateParameter();
	}

	Embedding<d*l,token,col> embedding;
	PositionalEncoder<d,l,col> positionalEncoder;
	DecoderLayer<d, l, col> layers1;
	DecoderLayer<d, l, col> layers2;
	DecoderLayer<d, l, col> layers3;
	DecoderLayer<d, l, col> layers4;
	DecoderLayer<d, l, col> layers5;
	DecoderLayer<d, l, col> layers6;
	LayerNorm<d * l,col> norm;
	Linear<d * l,col, vocab> linear;
	Softmax<d * l, vocab> softmax;

	Tensor& _input;
	Tensor& _output;
	Tensor& _inGradient;

	Tensor _out1;
	Tensor _out2;
	Tensor _out3_1;
	Tensor _out3_2;
	Tensor _out3_3;
	Tensor _out3_4;
	Tensor _out3_5;
	Tensor _out3_6;
	Tensor _out4;
	Tensor _out5;
	
	Tensor _gradient2;
	Tensor _gradient3_1;
	Tensor _gradient3_2;
	Tensor _gradient3_3;
	Tensor _gradient3_4;
	Tensor _gradient3_5;
	Tensor _gradient3_6;
	Tensor _gradient4;
	Tensor _gradient5;
};

#endif // !CLONE
