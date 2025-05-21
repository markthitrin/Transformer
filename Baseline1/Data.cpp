#include "Header.h"

Data::Data() noexcept { ; }

Data::Data(const std::vector<int>& tokens) noexcept :
	tokens(tokens) { ; }

std::pair<Tensor,Tensor> Data::getData(const int batch, const int seqeunceLEngth) noexcept {
	Tensor input(batch * sequenceLength, 1, 1);
	Tensor target(batch * sequenceLength, 1, 1);
	for (int i = 0; i < batch; i++) {
		int spos = RandomInt(0, tokens.size() - 1 - sequenceLength);
		for (int j = 0; j < sequenceLength - 1; j++) {
			input[i * sequenceLength + j][0][0] = tokens[spos + j];
			target[i * sequenceLength + j][0][0] = tokens[spos + j];
		}
		target[(i + 1) * sequenceLength - 1][0][0] = tokens[spos + sequenceLength - 1];
	}
	return std::make_pair(input, target);
}
