#ifndef DATA
#define DATA

#include "Header.h"

class Data {
public:
	Data() noexcept;
	Data(const std::vector<int>& tokens) noexcept;

	std::pair<Tensor,Tensor> getData(const int batch, const int seqeunceLEngth) noexcept;

	std::vector<int> tokens;
};

#endif 