#include "Header.h"

EncoderLayer::EncoderLayer() noexcept { ; }

EncoderLayer::EncoderLayer(const int dModel) noexcept :
	norm1(LayerNorm(dModel)),
	mulAtt(MultiheadAttention(head, dModel, qshape)),
	dropOut1(DropOut(dModel)),
	norm2(LayerNorm(dModel)),
	pff(PositionwiseFeedForward(dModel, dFF)),
	dropOut2(DropOut(dFF)) {
	;
}

Tensor EncoderLayer::operator()(const int sequenceLength, const Tensor& tensor, const Matrix& mask) noexcept {
	Tensor x = norm1(tensor);
	Tensor x1 = tensor + dropOut1(mulAtt(sequenceLength, x, x, x, mask));
	return x1 + dropOut2(pff(norm2(x1)));
}

Tensor EncoderLayer::predict(const int sequenceLength, const Tensor& tensor, const Matrix& mask) noexcept {
	Tensor x = norm1.predict(tensor);
	Tensor x1 = tensor + dropOut1.predict(mulAtt.predict(sequenceLength, x, x, x, mask));
	return x1 + dropOut2.predict(pff.predict(norm2.predict(x1)));
}


Tensor EncoderLayer::backpropagate(const int sequenceLength, const Tensor& gradient, const Matrix& mask) noexcept {
	Tensor x1Gradient = gradient + norm2.backpropagate(pff.backpropagate(dropOut2.backpropagate(gradient)));
	return x1Gradient + norm1.backpropagate(mulAtt.backpropagate(sequenceLength, dropOut1.backpropagate(x1Gradient), mask));
}

void EncoderLayer::updateParameter() noexcept {
	norm1.updateParameter();
	mulAtt.updateParameter();
	norm2.updateParameter();
	pff.updateParameter();
}

