#ifndef LAYER_NORM
#define LAYER_NORM

#include "Header.h"
#include "Tensor.h"
#include "Util.h"


template<int row,int col>
class LayerNorm {
public:
	LayerNorm() {
		_alpha.init();
		_bias.init();
		_xHat.init();
		_o.init();
		Set<1, col>(_alpha, 1.0f);
	}
	~LayerNorm() {
		_alpha.free();
		_bias.free();
		_xHat.free();
		_o.free();
	}

	void forward() noexcept {
		IMPORT_CONST(input);
		IMPORT_CONST(alpha);
		IMPORT_CONST(bias);
		IMPORT(o);
		IMPORT(xHat);
		IMPORT(output);

		for (int i = 0; i < row; i++) {
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
			o[i] /= (col - 1);
			o[i] = std::sqrt(o[i]);

			for (int j = 0; j < col; j++) {
				xHat[i * col + j] = (input[i * col + j] - mean) / (o[i] + eps);
				output[i * col + j] = alpha[j] * xHat[i * col + j] + bias[j];
			}
		}
	}

	void predict() noexcept {
		forward();
	}

	void backward() noexcept {
		IMPORT_CONST(inGradient);
		IMPORT_CONST(alpha);
		IMPORT_CONST(o);
		IMPORT_CONST(xHat);
		IMPORT(outGradient);
		IMPORTA(alphaGradient, _alphaOpt.gradient);
		IMPORTA(biasGradient, _biasOpt.gradient);

		feedCount++;
		constexpr float invCol = 1.0f / col;
		for (int i = 0; i < row; i++) {
			const float invO = 1.0f / (o[i] + eps);
			float sumG = 0;
			float sumGXHat = 0;
			for (int j = 0; j < col; j++) {
				float gxH = inGradient[i * col + j] * xHat[i * col + j];
				alphaGradient[j] += gxH;
				biasGradient[j] += inGradient[i * col + j];
				sumG += inGradient[i * col + j];
				sumGXHat += gxH;
			}
			float a = invCol * sumG;
			float b = invCol * sumGXHat;
			for (int j = 0; j < col; j++) {
				outGradient[i * col + j] = invO * (inGradient[i * col + j] - a - xHat[i * col + j] * b) * alpha[j];
			}
		}
	}

	void updateParameter() noexcept {
		AdamOpt(_alpha, _alphaOpt, feedCount);
		AdamOpt(_bias, _biasOpt, feedCount);

		feedCount = 0;
	}

	Tensor<row, col> _input;
	Tensor<row, col> _output;
	Tensor<row, col> _inGradient;
	Tensor<row, col> _outGradient;

	Tensor<1, col> _alpha;
	Tensor<1, col> _bias;

	int feedCount = 0;
	AdamOptGradient<1, col> _alphaOpt;
	AdamOptGradient<1, col> _biasOpt;

	Tensor<row, col> _xHat;
	Tensor<1, row> _o;
};

#endif