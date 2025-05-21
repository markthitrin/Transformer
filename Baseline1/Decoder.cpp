#include "Header.h"

Decoder::Decoder() { ; }

Decoder::Decoder(const int dModel, const int N, const int vocab) noexcept :
	embedding(Embedding(vocab, dModel)),
	positionalEncoder(PositionalEncoder()),
	layers(std::vector<DecoderLayer>(N, DecoderLayer(dModel))),
	norm(LayerNorm(dModel)),
	linear(Linear(dModel, vocab)) {
	;
}

Tensor Decoder::operator()(const int sequenceLength, const Tensor& tensor) noexcept {
	Tensor posX = positionalEncoder(sequenceLength, embedding(tensor));
	Tensor out = layers[0](sequenceLength, posX);
	for (int i = 1; i < layers.size(); i++) {
		out = layers[i](sequenceLength, out);
	}
	return softmax(linear(norm(out)));
}

Tensor Decoder::predict(const int sequenceLength, const Tensor& tensor) noexcept {
	Tensor posX = positionalEncoder.predict(sequenceLength, embedding.predict(tensor));
	Tensor out = layers[0].predict(sequenceLength, posX);
	for (int i = 1; i < layers.size(); i++) {
		out = layers[i].predict(sequenceLength, out);
	}
	return softmax.predict(linear.predict(norm.predict(out)));
}

Tensor Decoder::backpropagate(const int sequenceLength, const Tensor& gradient) noexcept {
	Tensor outGradient = norm.backpropagate(linear.backpropagate(softmax.backpropagate(gradient)));
	for (int i = layers.size() - 1; i >= 0; i--) {
		outGradient = layers[i].backpropagate(sequenceLength, outGradient);
	}
	return embedding.backpropagate(positionalEncoder.backpropagate(outGradient));
}

void Decoder::updateParameter() noexcept {
	embedding.updateParameter();
	for (int i = 0; i < layers.size(); i++) {
		layers[i].updateParameter();
	}
	norm.updateParameter();
	linear.updateParameter();
}