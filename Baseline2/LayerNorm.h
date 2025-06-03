#ifndef LAYER_NORM
#define LAYER_NORM

#include "Header.h"
#include "Tensor.h"
#include "Util.h"

template<int d,int col>
class LayerNorm {
public:
	LayerNorm() noexcept :
		_gamma			(Create<1,col>()),
		_bias			(Create0<1, col>()),
		_gammaGradient	(Create0<1, col>()),
		_gammaM			(Create0<1, col>()),
		_gammaV			(Create0<1, col>()),
		_biasGradient	(Create0<1, col>()),
		_biasM			(Create0<1, col>()),
		_biasV			(Create0<1, col>()),

		_xHat			(Create0<d, col>()),
		_o				(Create0<1, d>()) {
		
		Set<1, col>(_gamma, 1.0f);
	}
	~LayerNorm() {
		std::free(_gamma);
		std::free(_bias);
		std::free(_gammaGradient);
		std::free(_gammaM);
		std::free(_gammaV);
		std::free(_biasGradient);
		std::free(_biasM);
		std::free(_biasV);

		std::free(_xHat);
		std::free(_o);
	}

	void forward() noexcept {
		IMPORT_CONST(input);
		IMPORT_CONST(gamma);
		IMPORT_CONST(bias);
		IMPORT(o);
		IMPORT(xHat);
		IMPORT(output);
		for (int i = 0; i < d; i++) {
			float mean = 0.0f;
			for (int j = 0; j < col; j++) {
				mean += input[i * col + j];
			}
			mean /= col;

			o[i] = 0;
			for (int j = 0; j < col; j++) {
				const float x = (input[i * col + j] - mean);
				o[i] += x * x;
			}
			o[i] /= col;
			o[i] = std::sqrt(o[i]);

			for (int j = 0; j < col; j++) {
				xHat[i * col + j] = (input[i * col + j] - mean) / (o[i] + eps);
				output[i * col + j] = gamma[j] * xHat[i * col + j] + bias[j];
			}
		}
	}

	void predict() noexcept {
		forward();
	}

	void backpropagate() noexcept {
		IMPORT_CONST(inGradient);
		IMPORT_CONST(gamma);
		IMPORT_CONST(o);
		IMPORT_CONST(xHat);
		IMPORT(outGradient);
		IMPORT(gammaGradient);
		IMPORT(biasGradient);
		feedCount++;
		constexpr float invCol = 1.0f / col;
		for (int i = 0; i < d; i++) {
			const float invO = 1.0f / (o[i] + eps);
			float sumG = 0;
			float sumGXHat = 0;
			for (int j = 0; j < col; j++) {
				float gxH = inGradient[i * col + j] * xHat[i * col + j];
				gammaGradient[j] += gxH;
				biasGradient[j] += inGradient[i * col + j];
				sumG += inGradient[i * col + j];
				sumGXHat += gxH;
			}
			float a = invCol * sumG;
			float b = invCol * sumGXHat;
			for (int j = 0; j < col; j++) {
				outGradient[i * col + j] = invO * (inGradient[i * col + j] - a - xHat[i * col + j] * b) * gamma[j];
			}
		}
	}

	void updateParameter() noexcept {
		IMPORT(gamma);
		IMPORT(bias);
		IMPORT(gammaGradient);
		IMPORT(gammaM);
		IMPORT(gammaV);
		IMPORT(biasGradient);
		IMPORT(biasM);
		IMPORT(biasV);
		Div<1, col>(gammaGradient, feedCount, gammaGradient);
		Div<1, col>(biasGradient, feedCount, biasGradient);

		AdamOpt<1,col>((Tensor)gamma, (Tensor)gammaM, (Tensor)gammaV, (Tensor)gammaGradient, t);
		AdamOpt<1,col>((Tensor)bias, (Tensor)biasM, (Tensor)biasV, (Tensor)biasGradient, t);

		Reset<1,col>(gammaGradient);
		Reset<1,col>(biasGradient);
		feedCount = 0;
		t++;
	}

	Tensor _input;
	Tensor _output;
	Tensor _inGradient;
	Tensor _outGradient;

	Tensor _gamma;
	Tensor _bias;

	int t = 1;
	int feedCount = 0;
	Tensor _gammaGradient;
	Tensor _gammaM;
	Tensor _gammaV;
	Tensor _biasGradient;
	Tensor _biasM;
	Tensor _biasV;

	Tensor _xHat;
	Tensor _o;
};

#endif