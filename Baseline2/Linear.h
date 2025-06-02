#ifndef LINEAR
#define LINEAR

#include "Header.h"
#include "Tensor.h"
#include "Util.h"

template<int d,int in,int out>
class Linear {
public:
	Linear() noexcept :
		_weight			(Create<out,in>()), 
		_bias			(Create0<1, out>()),

		_weightGradient	(Create0<out,in>()),
		_weightM		(Create0<out,in>()),
		_weightV		(Create0<out,in>()), 
		_biasGradient	(Create0<1,out>()),
		_biasM			(Create0<1,out>()),
		_biasV			(Create0<1,out>()) {

		HeNormalInit<out, in, in>(_weight);
		HeNormalInit<1, out, out>(_bias);
	}
	~Linear() {
		_aligned_free(_weight);
		_aligned_free(_bias);

		_aligned_free(_weightGradient);
		_aligned_free(_weightM);
		_aligned_free(_weightV);
		_aligned_free(_biasGradient);
		_aligned_free(_biasM);
		_aligned_free(_biasV);
	}

	void forward() noexcept {
		IMPORT_CONST(input);
		IMPORT_CONST(weight);
		IMPORT_CONST(bias);
		IMPORT(output);
		Reset<d, out>(output);
		MatMulPlusABT<d, in, out>((Tensor)input, (Tensor)weight, (Tensor)output);
		for (int i = 0; i < d; i++) {
			Plus<1, out>((Tensor)output + out * i, (Tensor)bias, (Tensor)output + out * i);
		}
	}

	void predict() noexcept {
		forward();
	}

	void backpropagate() noexcept {
		IMPORT_CONST(inGradient);
		IMPORT_CONST(input);
		IMPORT_CONST(weight);
		IMPORT_CONST(bias);
		IMPORT(biasGradient);
		IMPORT(weightGradient);
		IMPORT(outGradient);
		feedCount++;
		Reset<d, in>(outGradient);
		for (int i = 0; i < d; i++) {
			Plus<1, out>((Tensor)biasGradient, (Tensor)inGradient + i * out, (Tensor)biasGradient);
		}

		MatMulPlusATB<out, d, in>((Tensor)inGradient, (Tensor)input, (Tensor)weightGradient);
		MatMulPlusAB<d, out, in>((Tensor)inGradient, (Tensor)weight, (Tensor)outGradient);
	}

	void updateParameter() noexcept {
		IMPORT(weight);
		IMPORT(bias);
		IMPORT(weightGradient);
		IMPORT(weightM);
		IMPORT(weightV);
		IMPORT(biasGradient);
		IMPORT(biasM);
		IMPORT(biasV);
		Div<out, in>(weightGradient, feedCount, weightGradient);
		Div<1, out>(biasGradient, feedCount, biasGradient);

		AdamOpt<out, in>(weight, weightM, weightV, weightGradient, t);
		AdamOpt<1, out>(bias, biasM, biasV, biasGradient, t);

		Reset<out, in>(weightGradient);
		Reset<1, out>(biasGradient);
		feedCount = 0;
		t++;
	}

	Tensor _input;
	Tensor _output;
	Tensor _inGradient;
	Tensor _outGradient;

	Tensor _weight;
	Tensor _bias;

	int t = 1;
	int feedCount = 0;
	Tensor _weightGradient;
	Tensor _weightM;
	Tensor _weightV;
	Tensor _biasGradient;
	Tensor _biasM;
	Tensor _biasV;
};

#endif // !LINEAR
