#ifndef EMBEDDING
#define EMBEDDING

#include "Header.h"
#include "Tensor.h"
#include "Util.h"

template<int d,int token,int col>
class Embedding {
public:
	Embedding() noexcept :
	_table			(Create<token, col>()),
	_tableGradient	(Create0<token, col>()),
	_tableM			(Create0<token, col>()),
	_tableV			(Create0<token, col>()),
	t				(Create<1, token>()) {

		UniformInit<token,col>(_table, 0.1f);
		Set<1,token>(t, 1.0f);
	}

	void forward() noexcept {
		IMPORT_CONST(input);
		IMPORT_CONST(table);
		IMPORT(output);
		for (int i = 0; i < d; i++) {
			Copy<1, col>(Tensor(table + int(input[i]) * col), Tensor(output + i * col));
		}
	}

	void predict() noexcept {
		forward();
	}

	void backpropagate() noexcept {
		feedCount++;
		IMPORT_CONST(input);
		IMPORT_CONST(inGradient);
		IMPORT(tableGradient);
		for (int i = 0; i < d; i++) {
			Plus<1, col>(Tensor(tableGradient + int(input[i]) * col), Tensor(inGradient + i * col), Tensor(tableGradient + int(input[i]) * col));
		}
	}

	void updateParameter() noexcept {
		IMPORT_CONST(input);
		IMPORT(table);
		IMPORT(tableGradient);
		IMPORT(tableM);
		IMPORT(tableV);
		/*Div<1, col>(Tensor(tableGradient), feedCount, Tensor(tableGradient));
		AdamOpt<1, col>(Tensor(table), Tensor(tableM), Tensor(tableV), Tensor(tableGradient), 1);
		Reset<token, col>(Tensor(tableGradient));*/

		for (int i = 0; i < d; i++) {
			int idx = input[i];
			Div<1, col>(Tensor(tableGradient + idx * col), feedCount, Tensor(tableGradient + idx * col));
			AdamOpt<1, col>(Tensor(table + idx * col),
							Tensor(tableM + idx * col),
							Tensor(tableV + idx * col),
							Tensor(tableGradient + idx * col),
							t[idx]);
			Reset<1, col>(Tensor(tableGradient + idx * col));
			t[idx] += 1;
		}
		feedCount = 0;
	}

	int feedCount = 0;
	Tensor t;

	Tensor _input;
	Tensor _output;
	Tensor _inGradient;

	Tensor _table;
	Tensor _tableGradient;
	Tensor _tableM;
	Tensor _tableV;
};

#endif
