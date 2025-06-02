#ifndef DATA
#define DATA

#include "Header.h"
#include "Tensor.h"

int RandomInt(int low, int high) {
	static std::random_device rd;
	static std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dist(low, high);
	return dist(gen);
}

class Data {
public:
	Data() noexcept {

	}

	Data(const std::vector<int>& tokens) noexcept :
		tokens(tokens) {
		;
	}

	template<int d, int l>
	void getData(Tensor input, Tensor target) noexcept {
		for (int i = 0; i < batch; i++) {
			int spos = RandomInt(0, tokens.size() - 1 - l);
			for (int j = 0; j < l - 1; j++) {
				input[i * l + j] = tokens[spos + j];
				target[i * l + j] = tokens[spos + j];
			}
			input[(i + 1) * l - 1] = 0;
			target[(i + 1) * l - 1] = tokens[spos + l - 1];
		}
	}

	std::vector<int> tokens;
};

#endif 