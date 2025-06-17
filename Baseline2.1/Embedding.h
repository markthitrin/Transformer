#ifndef EMBEDDING
#define EMBEDDING

#include "Header.h"
#include "Tensor.h"
#include "Util.h"

template<int row,int token,int col>
class Embedding {
public:
	Embedding() {
		for(int i = 0;i < token;i++) {
			_table[i].UniformInit(0.1f);
			feedCount[i] = 0;
		}
	}

	void forward() noexcept {
		IMPORT_CONST(input);

		const float sqrtCol = std::sqrt(col);
		for (int i = 0; i < row; i++) {
			Mul(_table[int(input[i])], sqrtCol, _output.template sliceRow<1>(i));
		}
	}

	void predict() noexcept {
		forward();
	}

	void backward() noexcept {
		IMPORT_CONST(input);
		IMPORT_CONST(inGradient);

		const float sqrtCol = std::sqrt(col);
		Mul(_inGradient, sqrtCol, _inGradient);
		for (int i = 0; i < row; i++) {
			feedCount[int(input[i])]++;
			Plus(_tableOpt[int(input[i])].gradient, _inGradient.template sliceRow<1>(i), _tableOpt[int(input[i])].gradient);
		}
	}

	void updateParameter() noexcept {
		for (int i = 0;i < token;i++) {
			AdamOpt(_table[i], _tableOpt[i], std::max(1, feedCount[i]));
			feedCount[i] = 0;
		}
	}

	void loadParam(cnpy::npz_t npFile, std::string prefix) {
		Tensor<token, col> loadRR;
		loadRR.init();
		loadRR.loadNp(npFile, prefix + ".weight");
		for (int i = 0;i < token;i++) {
			Copy(loadRR.template sliceRow<1>(i), _table[i]);
		}
	}

	void forwardTest(cnpy::npz_t npFile, std::string prefix) {
		_input.init();
		_output.init();
		Tensor<row, col> target;
		target.init();

		_input.loadNp(npFile, prefix + ".input");
		target.loadNp(npFile, prefix + ".output");

		forward();
		PrintTestResult("forward",_output, target);
	}

	void checkUpdatedParam(cnpy::npz_t npFile, std::string prefix) {
		Tensor<1, col> _tableUpdated[token];
		for(int i = 0;i < token;i ++) {
			_tableUpdated[i].init();
		}
		Tensor<token, col> loadRR;
		loadRR.init();
		loadRR.loadNp(npFile, prefix + ".updated_weights");
		for (int i = 0;i < token;i++) {
			Copy(loadRR.template sliceRow<1>(i), _tableUpdated[i]);
		}

		for(int i = 0;i < token;i++) {
			PrintTestResult("backward table:" + std::to_string(i), _table[i], _tableUpdated[i]);
		}
	}

	void backwardTest(cnpy::npz_t npFile, std::string prefix) {
		_output.init();
		_inGradient.init();
		Set(_inGradient,1.0f / row / col);
		_input.init();
		_input.loadNp(npFile, prefix + ".input");

		forward();
		backward();
		updateParameter();

		checkUpdatedParam(npFile, prefix);
	}

	Tensor<1, row> _input;
	Tensor<row, col> _output;
	Tensor<row, col> _inGradient;

	int feedCount[token];
	Tensor<1, col> _table[token];
	AdamOptGradient<1, col> _tableOpt[token];
};

#endif
