#include "Header.h"

DecoderLayer::DecoderLayer() noexcept { ; }

DecoderLayer::DecoderLayer(const int dModel) noexcept :
	norm1(LayerNorm(dModel)),
	mulAtt(MultiheadAttention(head, dModel, qshape)),
	dropOut1(DropOut()),
	norm2(LayerNorm(dModel)),
	pff(PositionwiseFeedForward(dModel, dFF)),
	dropOut2(DropOut()) {
	;
}

Tensor DecoderLayer::operator()(const int sequenceLength, const Tensor& tensor) noexcept {
	Tensor x = norm1(tensor);
	Tensor x1 = tensor + dropOut1(mulAtt(sequenceLength, x, x, x, Matrix::lookAheadMask(sequenceLength)));
	return x1 + dropOut2(pff(norm2(x1)));
}

Tensor DecoderLayer::predict(const int sequenceLength, const Tensor& tensor) noexcept {
	Tensor x = norm1.predict(tensor);
	Tensor x1 = tensor + dropOut1.predict(mulAtt.predict(sequenceLength, x, x, x, Matrix::lookAheadMask(sequenceLength)));
	return x1 + dropOut2.predict(pff.predict(norm2.predict(x1)));
}


Tensor DecoderLayer::backpropagate(const int sequenceLength, const Tensor& gradient) noexcept {
	Tensor x1Gradient = gradient + norm2.backpropagate(pff.backpropagate(dropOut2.backpropagate(gradient)));
	return x1Gradient + norm1.backpropagate(mulAtt.backpropagate(sequenceLength, dropOut1.backpropagate(x1Gradient), Matrix::lookAheadMask(sequenceLength)));
}

void DecoderLayer::updateParameter() noexcept {
	norm1.updateParameter();
	mulAtt.updateParameter();
	norm2.updateParameter();
	pff.updateParameter();
}

