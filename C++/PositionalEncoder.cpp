#include "Config.h"
#include "DropOut.h"
#include "Header.h"
#include "PositionalEncoder.h"
#include "Tensor.h"
#include "Timer.h"
#include "Util.h"

PositionalEncoder::PositionalEncoder() :
    dropout(batch * sequenceLength, dModel),
    
    positionEncode(sequenceLength, dModel) {

    GetPositionalEncode(positionEncode);
}

void PositionalEncoder::forward(TensorView input, TensorView output) {
    for(int i = 0;i < batch;i++) {
        Plus(input.sliceRow(i * sequenceLength, sequenceLength), positionEncode, input.sliceRow(i * sequenceLength, sequenceLength));
    }
    Timer::CheckPoint();
    dropout.forward(input, output);
}

void PositionalEncoder::predict(TensorView input, TensorView output) {
    for(int i = 0;i < batch;i++) {
        Plus(input.sliceRow(i * sequenceLength, sequenceLength), positionEncode, output.sliceRow(i * sequenceLength, sequenceLength));
    }
}

void PositionalEncoder::backward(TensorView outputGradient, TensorView inputGradient) {
    dropout.backward(outputGradient, inputGradient);
    Timer::CheckPoint();
}