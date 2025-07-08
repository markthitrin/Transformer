

#include "Header.h"
#include "Tensor.h"
#include "Util.h"
#include "cnpy.h"
#include "DropOut.h"
#include "PositionalEncoder.h"

PositionalEncoder::PositionalEncoder() :
    dropout(batch * sequenceLength, dModel) {

    GetPositionalEncode(positionEncode);
}

void PositionalEncoder::forward(TensorView input, TensorView output) {
    for(int i = 0;i < batch;i++) {
        Plus(input.sliceRow(i * sequenceLength, sequenceLength), positionEncode, input.sliceRow(i * sequenceLength, sequenceLength));
    }
    dropout.forward(input, output);
}

void PositionalEncoder::predict(TensorView input, TensorView output) {
    for(int i = 0;i < batch;i++) {
        Plus(input.sliceRow(i * sequenceLength, sequenceLength), positionEncode, output.sliceRow(i * sequenceLength, sequenceLength));
    }
}

void PositionalEncoder::backward(TensorView outputGradient, TensorView inputGradient) {
    dropout.backward(outputGradient, inputGradient);
}