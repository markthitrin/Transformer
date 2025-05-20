#include "Header.h"

Linear::Linear() { ; }

Linear::Linear(const int inshape, const int outshape) : 
	inshape(inshape), 
	outshape(outshape),
	weight(Matrix(outshape,inshape)),
	bias(Matrix(outshape,1)),
	weightGradient(Matrix(outshape,inshape)),
	weightM(Matrix(outshape,inshape)),
	weightV(Matrix(outshape, inshape)),
	biasGradient(Matrix(outshape, 1)),
	biasM(Matrix(outshape, 1)),
	biasV(Matrix(outshape, 1)) { 
	HeNormalInit(weight);
	HeNormalInit(bias);
}

Tensor Linear::operator()(const Tensor& tensor) noexcept {
	Tensor result(tensor.batch, outshape, 1);
	x = tensor;
	for (int i = 0; i < tensor.batch; i++) {
		result[i] = weight * tensor.data[i] + bias;
	}
	return result;
}

Tensor Linear::predict(const Tensor& tensor) noexcept {
	Tensor result(tensor.batch, outshape, 1);
	x = tensor;
	for (int i = 0; i < tensor.batch; i++) {
		result[i] = weight * tensor.data[i] + bias;
	}
	return result;
}


Tensor Linear::backpropagate(const Tensor& gradient) noexcept {
	feedCount++;
	Tensor result(gradient.batch, inshape, 1);
	for (int i = 0; i < gradient.batch; i++) {
		for (int j = 0; j < outshape; j++) {
			for (int k = 0; k < inshape; k++) {
				weightGradient[j][k] += gradient[i][j][0] * x[i][k][0];
				result[i][k][0] += gradient[i][j][0] * weight[j][k];
			}
		}
		for (int j = 0; j < outshape; j++) {
			biasGradient[j][0] += gradient[i][j][0];
		}
	}
	return result;
}

void Linear::updateParameter() noexcept {
	weightGradient /= feedCount;
	biasGradient /= feedCount;
	weight -= AdamOpt(weightM, weightV, weightGradient, t);
	bias -= AdamOpt(biasM, biasV, biasGradient, t);

	weightGradient = 0;
	biasGradient = 0;
	feedCount = 0;
	t++;
}

