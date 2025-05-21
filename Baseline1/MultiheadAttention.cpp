#include "Header.h"

static std::vector<Tensor> decompose(const Tensor& tensor, const int num) {
	std::vector<Tensor> tensors(num, Tensor(tensor.batch, tensor.data[0].row / num, 1));
	for (int i = 0; i < tensor.batch; i++) {
		for (int j = 0; j < num; j++) {
			for (int k = 0; k < tensor.data[0].row / num; k++) {
				tensors[j][i][k][0] = tensor[i][j * tensor.data[0].row / num + k][0];
			}
		}
	}
	return tensors;
}

MultiheadAttention::MultiheadAttention() noexcept { ; }

MultiheadAttention::MultiheadAttention(const int head, const int dModel, const int qshape) noexcept :
	head(head),
	dModel(dModel),
	qshape(qshape),
	vshape(dModel / head),
	attLayer(std::vector<Attention>(head, Attention(dModel / head, qshape, dModel / head))),
	WO(Matrix(dModel, dModel)),
	WOGradient(Matrix(dModel, dModel)),
	WOM(Matrix(dModel, dModel)),
	WOV(Matrix(dModel, dModel)) { 
	XavierUniformInit(WO);
}

Tensor MultiheadAttention::operator()(const int sequenceLength, const Tensor& tensorQ, const Tensor& tensorK, const Tensor& tensorV, const Matrix& mask) noexcept {
	Tensor result(tensorQ.batch, tensorQ.data[0].row, 1);
	std::vector<Tensor> tensorsQ = decompose(tensorQ, head);
	std::vector<Tensor> tensorsK = decompose(tensorK, head);
	std::vector<Tensor> tensorsV = decompose(tensorV, head);
	Tensor Attout(tensorQ.batch, head * vshape, 1);
	for (int i = 0; i < head; i++) {
		Tensor out = attLayer[i](sequenceLength, tensorsQ[i], tensorsK[i], tensorsV[i], mask);
		const int s = i * vshape;
		for (int j = 0; j < out.batch; j++) {
			for (int k = 0; k < vshape; k++) {
				Attout[j][s + k][0] = out[j][k][0];
			}
		}
	}
	for (int i = 0; i < tensorQ.batch; i++) {
		result[i] = WO * Attout[i];
	}
	this->Attout = Attout;
	return result;
}

Tensor MultiheadAttention::predict(const int sequenceLength, const Tensor& tensorQ, const Tensor& tensorK, const Tensor& tensorV, const Matrix& mask) noexcept {
	Tensor result(tensorQ.batch, tensorQ.data[0].row, 1);
	std::vector<Tensor> tensorsQ = decompose(tensorQ, head);
	std::vector<Tensor> tensorsK = decompose(tensorK, head);
	std::vector<Tensor> tensorsV = decompose(tensorV, head);
	Tensor Attout(tensorQ.batch, head * vshape, 1);
	for (int i = 0; i < head; i++) {
		Tensor out = attLayer[i].predict(sequenceLength, tensorsQ[i], tensorsK[i], tensorsV[i], mask);
		const int s = i * vshape;
		for (int j = 0; j < out.batch; j++) {
			for (int k = 0; k < vshape; k++) {
				Attout[j][s + k][0] = out[j][k][0];
			}
		}
	}
	for (int i = 0; i < tensorQ.batch; i++) {
		result[i] = WO * Attout[i];
	}
	this->Attout = Attout;
	return result;
}


Tensor MultiheadAttention::backpropagate(const int sequenceLength, const Tensor& gradient, const Matrix& mask) noexcept {
	feedCount++;
	Tensor result(gradient.batch, dModel, 1);
	Tensor AttoutGradient(Attout.batch, Attout.data[0].row, 1);
	for (int i = 0; i < gradient.batch; i++) {
		for (int j = 0; j < dModel; j++) {
			for (int k = 0; k < dModel; k++) {
				WOGradient[j][k] += gradient[i][j][0] * Attout[i][k][0];
				AttoutGradient[i][k][0] += gradient[i][j][0] * WO[j][k];
			}
		}
	}
	std::vector<Tensor> AttGradients = decompose(AttoutGradient, head);
	for (int i = 0; i < head; i++) {
		Tensor out = attLayer[i].backpropagate(sequenceLength, AttGradients[i], mask);
		const int s = i * vshape;
		for (int j = 0; j < out.batch; j++) {
			for (int k = 0; k < vshape; k++) {
				result[j][s + k][0] = out[j][k][0];
			}
		}
	}
	return result;
}

void MultiheadAttention::updateParameter() noexcept {
	for (int i = 0; i < head; i++) {
		attLayer[i].updateParameter();
	}
	WOGradient /= feedCount;
	WO -= AdamOpt(WOM, WOV, WOGradient, t);

	WOGradient = 0;
	feedCount = 0;
	t++;
}