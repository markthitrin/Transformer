#ifndef LINEAR
#define LINEAR

#include "Header.h"
#include "Tensor.h"
#include "Util.h"

template<int row,int in,int out>
class Linear {
public:
	Linear() noexcept {
		_weight.HeNormalInit();
		_bias.HeNormalInit();
	}

	void forward() noexcept {
		Reset(_output);
		MatMulPlusABT(_input, _weight, _output);
		for (int i = 0; i < row; i++) {
			Plus(_output.template sliceRow<1>(i), _bias, _output.template sliceRow<1>(i));
		}
	}

	void predict() noexcept {
		forward();
	}

	void backward() noexcept {
		feedCount++;
		Reset(_outGradient);
		for (int i = 0; i < row; i++) {
			Plus(_biasOpt.gradient, _inGradient.template sliceRow<1>(i), _biasOpt.gradient);
		}

		MatMulPlusATB(_inGradient, _input, _weightOpt.gradient);
		MatMulPlusAB(_inGradient, _weight, _outGradient);
	}

	void updateParameter() noexcept {
		AdamOpt(_weight, _weightOpt, feedCount);
		AdamOpt(_bias, _biasOpt, feedCount);
		feedCount = 0;
	}

	void checkUpdatedParam(cnpy::npz_t npFile, std::string prefix) {
		Tensor<tgtVocab, dModel> weightUpdated;
		Tensor<1, tgtVocab> biasUpdated;
		weightUpdated.init();
		biasUpdated.init();
		weightUpdated.loadNp(npFile, prefix + ".updated_weight");
		biasUpdated.loadNp(npFile, prefix + ".updated_bias");

		PrintTestResult("backward " + prefix + ".weight", _weight, weightUpdated);
		PrintTestResult("backward " + prefix + ".bias", _bias, biasUpdated);
	}

	Tensor<row, in> _input;
	Tensor<row, out> _output;
	Tensor<row, out> _inGradient;
	Tensor<row, in> _outGradient;

	Tensor<out, in> _weight;
	Tensor<1, out> _bias;

	int feedCount = 0;
	AdamOptGradient<out, in> _weightOpt;
	AdamOptGradient<1, out> _biasOpt;
};

#endif // !LINEAR
