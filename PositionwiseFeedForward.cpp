#include "Header.h"

PositionwiseFeedForward::PositionwiseFeedForward() noexcept { ; }

PositionwiseFeedForward::PositionwiseFeedForward(const int dModel, const int dFF)noexcept :
	linear1(Linear(dModel,dFF)),
	relu(ReLU()),
	dropout(DropOut()),
	linear2(Linear(dFF,dModel))
{ ; }

Tensor PositionwiseFeedForward::operator()(const Tensor& tensor) noexcept {
	return linear2(
		dropout(
			relu(
				linear1(
					tensor
				))));
}

Tensor PositionwiseFeedForward::predict(const Tensor& tensor) noexcept {
	return linear2.predict(
		dropout.predict(
			relu.predict(
				linear1.predict(
					tensor
				))));
}


Tensor PositionwiseFeedForward::backpropagate(const Tensor& gradient) noexcept {
	return linear1.backpropagate(
		relu.backpropagate(
			dropout.backpropagate(
				linear2.backpropagate(
					gradient
				))));
}

void PositionwiseFeedForward::updateParameter() noexcept {
	linear1.updateParameter();
	linear2.updateParameter();
}
