#include "Header.h"

LayerNorm::LayerNorm() noexcept { ; }

LayerNorm::LayerNorm(const int shape) noexcept :
	shape(shape),
	y(Matrix(shape, 1)),
	b(Matrix(shape, 1)),
	yGradient(Matrix(shape, 1)),
	yM(Matrix(shape, 1)),
	yV(Matrix(shape, 1)),
	bGradient(Matrix(shape, 1)),
	bM(Matrix(shape, 1)),
	bV(Matrix(shape, 1)){
	y = Matrix(shape, 1, 1);
	b = Matrix(shape, 1, 0);
}

Tensor LayerNorm::operator()(const Tensor& tensor) noexcept {
	Tensor result(tensor.batch, tensor.data[0].row, 1);
	o2 = Tensor(tensor.batch, 1, 1);
	xHat = Tensor(tensor.batch, tensor.data[0].row, 1);
	for (int i = 0; i < tensor.batch; i++) {
		float u = 0;
		for (int j = 0; j < tensor.data[0].row; j++) {
			u += tensor.data[i][j][0];
		}
		u /= tensor.data[0].row;
		for (int j = 0; j < tensor.data[0].row; j++) {
			o2[i][0][0] += std::pow(tensor.data[i][j][0] - u, 2.0f);
		}
		o2[i][0][0] /= tensor.data[0].row;
		for (int j = 0; j < tensor.data[0].row; j++) {
			xHat[i][j][0] = (tensor[i][j][0] - u) / std::sqrt(o2[i][0][0] + eps);
			result[i][j][0] = y[j][0] * xHat[i][j][0] + b[j][0];
		}
	}
	return result;
}

Tensor LayerNorm::predict(const Tensor& tensor) noexcept {
	Tensor result(tensor.batch, tensor.data[0].row, 1);
	o2 = Tensor(tensor.batch, 1, 1);
	xHat = Tensor(tensor.batch, tensor.data[0].row, 1);
	for (int i = 0; i < tensor.batch; i++) {
		float u = 0;
		for (int j = 0; j < tensor.data[0].row; j++) {
			u += tensor.data[i][j][0];
		}
		u /= tensor.data[0].row;
		for (int j = 0; j < tensor.data[0].row; j++) {
			o2[i][0][0] += std::pow(tensor.data[i][j][0] - u, 2.0f);
		}
		o2[i][0][0] /= tensor.data[0].row;
		for (int j = 0; j < tensor.data[0].row; j++) {
			xHat[i][j][0] = (tensor[i][j][0] - u) / std::sqrt(o2[i][0][0] + eps);
			result[i][j][0] = y[j][0] * xHat[i][j][0] + b[j][0];
		}
	}
	return result;
}


Tensor LayerNorm::backpropagate(const Tensor& gradient) noexcept {
	feedCount++;
	Tensor result(gradient.batch, gradient.data[0].row, 1);
	for (int i = 0; i < gradient.batch; i++) {
		for (int j = 0; j < shape; j++) {
			yGradient[j][0] += gradient[i][j][0] * xHat[i][j][0];
			bGradient[j][0] += gradient[i][j][0];
		}
		float invO2 = (1.0f) / std::sqrt(o2[i][0][0] + eps);
		float sumD = 0;
		float sumDXHat = 0;
		for (int j = 0; j < shape; j++) {
			sumD += gradient[i][j][0];
			sumDXHat += gradient[i][j][0] * xHat[i][j][0];
		}
		for (int j = 0; j < shape; j++) {
			result[i][j][0] = invO2 * (gradient[i][j][0] - 1.0f / shape * sumD - xHat[i][j][0] / shape * sumDXHat) * y[j][0];
		}
	}
	return result;
}

void LayerNorm::updateParameter() noexcept {
	yGradient /= feedCount;
	bGradient /= feedCount;
	y -= AdamOpt(yM, yV, yGradient, t);
	b -= AdamOpt(bM, bV, bGradient, t);

	yGradient = 0;
	bGradient = 0;
	feedCount = 0;
	t++;
}