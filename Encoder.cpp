#include "Header.h"

Encoder::Encoder() { ; }

Encoder::Encoder(const int dModel, const int N) noexcept :
	layers(std::vector<EncoderLayer>(N,EncoderLayer(dModel))),
	norm(LayerNorm(dModel))
{ ; }

Tensor Encoder::operator()(const int sequenceLength, const Tensor& tensor) noexcept {
	Tensor out = layers[0](sequenceLength, tensor, Matrix(sequenceLength,sequenceLength,1));
	for (int i = 1; i < layers.size(); i++) {
		out = layers[i](sequenceLength, out, Matrix(sequenceLength, sequenceLength, 1));
	}
	return norm(out);
}

Tensor Encoder::predict(const int sequenceLength, const Tensor& tensor) noexcept {
	Tensor out = layers[0].predict(sequenceLength, tensor, Matrix(sequenceLength, sequenceLength, 1));
	for (int i = 1; i < layers.size(); i++) {
		out = layers[i].predict(sequenceLength, out, Matrix(sequenceLength, sequenceLength, 1));
	}
	return norm.predict(out);
}


Tensor Encoder::backpropagate(const int sequenceLength, const Tensor& gradient) noexcept {
	Tensor outGradient = layers.back().backpropagate(sequenceLength, gradient, Matrix(sequenceLength, sequenceLength, 1));
	for (int i = layers.size() - 2; i >= 0; i--) {
		outGradient = layers[i].backpropagate(sequenceLength, outGradient, Matrix(sequenceLength, sequenceLength, 1));
	}
	return outGradient;
}

void Encoder::updateParameter() noexcept {
	for (int i = 0; i < layers.size(); i++) {
		layers[i].updateParameter();
	}
	norm.updateParameter();
}