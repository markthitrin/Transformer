#ifndef POSITIONWISE_FEED_FORWARD
#define POSITIONWISE_FEED_FORWARD

#include "Header.h"

class PositionwiseFeedForward {
public:
	PositionwiseFeedForward() noexcept;
	PositionwiseFeedForward(const int dModel, const int dFF) noexcept;

	Tensor operator()(const Tensor& tensor) noexcept;
	Tensor predict(const Tensor& tensor) noexcept;
	Tensor backpropagate(const Tensor& gradient) noexcept;
	void updateParameter() noexcept;

	Linear linear1	= Linear();
	ReLU relu		= ReLU();
	DropOut dropout = DropOut();
	Linear linear2	= Linear();
};

#endif