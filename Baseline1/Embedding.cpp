#include "Header.h"

Embedding::Embedding() { ; }

Embedding::Embedding(const int numtoken, const int outshape) noexcept : 
	numToken(numtoken), 
	outshape(outshape),
	table(std::vector<Matrix>(numtoken, Matrix(outshape, 1))),
	tableGradient(std::vector<Matrix>(numtoken, Matrix(outshape, 1))),
	tableM(std::vector<Matrix>(numtoken, Matrix(outshape, 1))),
	tableV(std::vector<Matrix>(numtoken, Matrix(outshape, 1))),
	used(std::vector<bool>(numtoken, false)) { 
	for (int i = 0; i < table.size(); i++) {
		UniformInit(table[i], 0.1);
	}
}

Tensor Embedding::operator()(const Tensor& tensor) noexcept {
	feedCount++;
	Tensor result(tensor.batch, outshape, 1);
	for (int i = 0; i < tensor.batch; i++) {
		result[i] = table[int(tensor[i][0][0])];
	}
	x = tensor;
	return result;
}

Tensor Embedding::predict(const Tensor& tensor) noexcept {
	feedCount++;
	Tensor result(tensor.batch, outshape, 1);
	for (int i = 0; i < tensor.batch; i++) {
		result[i] = table[int(tensor[i][0][0])];
	}
	x = tensor;
	return result;
}


Tensor Embedding::backpropagate(const Tensor& gradient) noexcept {
	for (int i = 0; i < gradient.batch; i++) {
		for (int j = 0; j < outshape; j++) {
			tableGradient[x[i][0][0]][j][0] += gradient[i][j][0];
			used[x[i][0][0]] = true;
		}
	}
	return Tensor(0, 0, 0); // The first layer of all;
}

void Embedding::updateParameter() noexcept {
	for (int i = 0; i < numToken; i++) {
		if (used[i]) {
			tableGradient[i] /= feedCount;
			table[i] -= AdamOpt(tableM[i], tableV[i], tableGradient[i], t);

			tableGradient[i] = 0;
			used[i] = false;
		}
	}
	feedCount = 0;
	t++;
}