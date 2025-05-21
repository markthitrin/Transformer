#include "Header.h"

Attention::Attention() noexcept { ; }

Attention::Attention(const int inshape, const int qshape, const int outshape) noexcept :
	inshape(inshape),
	qshape(qshape),
	outshape(outshape),
	WQ(Matrix(qshape, inshape)),
	WK(Matrix(qshape, inshape)),
	WV(Matrix(outshape, inshape)),
	WQGradient(Matrix(qshape, inshape)),
	WKGradient(Matrix(qshape, inshape)),
	WVGradient(Matrix(outshape, inshape)),
	WQM(Matrix(qshape, inshape)),
	WKM(Matrix(qshape, inshape)),
	WVM(Matrix(outshape, inshape)),
	WQV(Matrix(qshape, inshape)),
	WKV(Matrix(qshape, inshape)),
	WVV(Matrix(outshape, inshape)) {
	XavierUniformInit(WQ);
	XavierUniformInit(WK);
	XavierUniformInit(WV);
}

Tensor Attention::operator()(const int sequenceLength, const Tensor& tensorQ, const Tensor& tensorK, const Tensor& tensorV, const Matrix& mask) noexcept {
	Tensor result(tensorQ.batch, outshape, 1);
	auto MatrixMul = [&](const Matrix& m, const Tensor& tensor) {
		Tensor result(tensor.batch, m.row, 1);
		for (int i = 0; i < tensor.batch; i++) {
			result[i] = m * tensor[i];
		}
		return result;
		};

	const int batch = tensorQ.batch / sequenceLength;
	Tensor Q = MatrixMul(WQ, tensorQ);
	Tensor K = MatrixMul(WK, tensorK);
	Tensor V = MatrixMul(WV, tensorV);
	Tensor S(batch, sequenceLength, sequenceLength);
	Tensor A(batch, sequenceLength, sequenceLength);
	for (int i = 0; i < batch; i++) {
		const int s = sequenceLength * i;
		for (int j = 0; j < sequenceLength; j++) {
			for (int k = 0; k < qshape; k++) {
				for (int w = 0; w < sequenceLength; w++) {
					S[i][j][w] += Q[s + j][k][0] * K[s + w][k][0] / std::sqrt(qshape);
				}
			}
		}
	}
	for (int i = 0; i < batch; i++) {
		for (int j = 0; j < sequenceLength; j++) {
			float sumExp = 0.0;
			float maxValue = -FLT_MAX;
			for (int k = 0; k < sequenceLength; k++) {
				if (mask[j][k] == 0) S[i][j][k] = 1e-9;
				maxValue = std::max(S[i][j][k], maxValue);
			}
			for (int k = 0; k < sequenceLength; k++) {
				sumExp += std::exp(S[i][j][k] - maxValue);
			}
			for (int k = 0; k < sequenceLength; k++) {
				A[i][j][k] = std::exp(S[i][j][k] - maxValue) / sumExp;
			}
		}
	}
	for (int i = 0; i < batch; i++) {
		const int s = sequenceLength * i;
		for (int j = 0; j < sequenceLength; j++) {
			for (int k = 0; k < sequenceLength; k++) {
				for (int w = 0; w < outshape; w++) {
					result[s + j][w][0] += A[i][j][k] * V[s + k][w][0];
				}
			}
		}
	}
	this->Q = Q;
	this->K = K;
	this->V = V;
	this->S = S;
	this->A = A;
	this->xQ = tensorQ;
	this->xK = tensorK;
	this->xV = tensorV;
	return result;
}

Tensor Attention::predict(const int sequenceLength, const Tensor& tensorQ, const Tensor& tensorK, const Tensor& tensorV, const Matrix& mask) noexcept {
	Tensor result(tensorQ.batch, outshape, 1);
	auto MatrixMul = [&](const Matrix& m, const Tensor& tensor) {
		Tensor result(tensor.batch, m.row, 1);
		for (int i = 0; i < tensor.batch; i++) {
			result[i] = m * tensor[i];
		}
		return result;
		};

	const int batch = tensorQ.batch / sequenceLength;
	Tensor Q = MatrixMul(WQ, tensorQ);
	Tensor K = MatrixMul(WK, tensorK);
	Tensor V = MatrixMul(WV, tensorV);
	Tensor S(batch, sequenceLength, sequenceLength);
	Tensor A(batch, sequenceLength, sequenceLength);
	for (int i = 0; i < batch; i++) {
		const int s = sequenceLength * i;
		for (int j = 0; j < sequenceLength; j++) {
			for (int k = 0; k < qshape; k++) {
				for (int w = 0; w < sequenceLength; w++) {
					S[i][j][w] += Q[s + j][k][0] * K[s + w][k][0] / std::sqrt(qshape);
				}
			}
		}
	}
	for (int i = 0; i < batch; i++) {
		for (int j = 0; j < sequenceLength; j++) {
			float sumExp = 0.0;
			float maxValue = -FLT_MAX;
			for (int k = 0; k < sequenceLength; k++) {
				if (mask[j][k] == 0) S[i][j][k] = 1e-9;
				maxValue = std::max(S[i][j][k], maxValue);
			}
			for (int k = 0; k < sequenceLength; k++) {
				sumExp += std::exp(S[i][j][k] - maxValue);
			}
			for (int k = 0; k < sequenceLength; k++) {
				A[i][j][k] = std::exp(S[i][j][k] - maxValue) / sumExp;
			}
		}
	}
	for (int i = 0; i < batch; i++) {
		const int s = sequenceLength * i;
		for (int j = 0; j < sequenceLength; j++) {
			for (int k = 0; k < sequenceLength; k++) {
				for (int w = 0; w < outshape; w++) {
					result[s + j][w][0] += A[i][j][k] * V[s + k][w][0];
				}
			}
		}
	}
	this->Q = Q;
	this->K = K;
	this->V = V;
	this->S = S;
	this->A = A;
	this->xQ = tensorQ;
	this->xK = tensorK;
	this->xV = tensorV;
	return result;
}

Tensor Attention::backpropagate(const int sequenceLength, const Tensor& gradient, const Matrix& mask) noexcept {
	feedCount++;
	Tensor result(gradient.batch, inshape, 1);
	const int batch = gradient.batch / sequenceLength;
	Tensor AGradient(batch, sequenceLength, sequenceLength);
	Tensor VGradient(batch * sequenceLength, outshape, 1);
	Tensor SGradient(batch, sequenceLength, sequenceLength);
	Tensor QGradient(batch * sequenceLength, qshape, 1);
	Tensor KGradient(batch * sequenceLength, qshape, 1);
	for (int i = 0; i < batch; i++) {
		const int s = sequenceLength * i;
		for (int j = 0; j < sequenceLength; j++) {
			for (int k = 0; k < sequenceLength; k++) {
				for (int w = 0; w < outshape; w++) {
					AGradient[i][j][k] += gradient[s + j][w][0] * V[s + k][w][0];
					VGradient[s + k][w][0] += gradient[s + j][w][0] * A[i][j][k];
				}
			}
		}
	}
	for (int i = 0; i < batch; i++) {
		const int s = sequenceLength * i;
		for (int j = 0; j < sequenceLength; j++) {
			for (int k = 0; k < sequenceLength; k++) {
				for (int w = 0; w < sequenceLength; w++) {
					SGradient[i][j][k] += AGradient[i][j][w] * A[i][j][w] * (float(w == k) - A[i][j][k]) * mask[j][k];
				}
			}
		}
	}
	for (int i = 0; i < batch; i++) {
		const int s = sequenceLength * i;
		for (int j = 0; j < sequenceLength; j++) {
			for (int k = 0; k < qshape; k++) {
				for (int w = 0; w < sequenceLength; w++) {
					QGradient[s + j][k][0] += SGradient[i][j][w] * K[s + w][k][0] / std::sqrt(qshape);
					KGradient[s + w][k][0] += SGradient[i][j][w] * Q[s + j][k][0] / std::sqrt(qshape);
				}
			}
		}
	}
	for (int i = 0; i < gradient.batch; i++) {
		for (int j = 0; j < qshape; j++) {
			for (int k = 0; k < inshape; k++) {
				WQGradient[j][k] += xQ[i][k][0] * QGradient[i][j][0];
				WKGradient[j][k] += xK[i][k][0] * KGradient[i][j][0];
			}
		}
		for (int j = 0; j < outshape; j++) {
			for (int k = 0; k < inshape; k++) {
				WVGradient[j][k] += xV[i][k][0] * VGradient[i][j][0];
			}
		}
	}
	for (int i = 0; i < gradient.batch; i++) {
		for (int j = 0; j < inshape; j++) {
			for (int k = 0; k < qshape; k++) {
				result[i][j][0] += WQ[k][j] * QGradient[i][k][0];
				result[i][j][0] += WK[k][j] * KGradient[i][k][0];
				
			}
			for (int k = 0; k < outshape; k++) {
				result[i][j][0] += WV[k][j] * VGradient[i][k][0];
			}
		}
	}
	return result;
}

void Attention::updateParameter() noexcept {
	WQGradient /= feedCount;
	WKGradient /= feedCount;
	WVGradient /= feedCount;
	WQ -= AdamOpt(WQM, WQV, WQGradient, t);
	WK -= AdamOpt(WKM, WKV, WKGradient, t);
	WV -= AdamOpt(WVM, WVV, WVGradient, t);

	WQGradient = 0;
	WKGradient = 0;
	WVGradient = 0;
	feedCount = 0;
	t++;
}