#ifndef POSITIONAL_ENCODER
#define POSITIONAL_ENCODER

#include "Header.h"
#include "Tensor.h"
#include "Util.h"

template<int d,int l,int col>
class PositionalEncoder {
public:
	PositionalEncoder() noexcept :
		_positionEncode(Create<d * l, col>()) {
		GetPositionalEncode<d,l,col>(_positionEncode);
	}

	void forward() noexcept {
		IMPORT_CONST(input);
		IMPORT_CONST(positionEncode);
		IMPORT(output);
		Plus<d* l, col>((Tensor)input, (Tensor)positionEncode, (Tensor)output);
	}

	void predict() noexcept {
		forward();
	}

	Tensor _input;
	Tensor _output;

	Tensor _positionEncode;
};

#endif
