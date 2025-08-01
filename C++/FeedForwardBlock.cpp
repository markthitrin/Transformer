#include "Config.h"
#include "DropOut.h"
#include "FeedForwardBlock.h"
#include "Header.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "ReLU.h"
#include "Tensor.h"
#include "Util.h"

FeedForwardBlock::FeedForwardBlock() :
    linear1(dModel, dFF),
    dropout(batch * sequenceLength, dFF),
    linear2(dFF, dModel),
    
    out1(batch * sequenceLength, dFF),
    out2(batch * sequenceLength, dFF),
    out3(batch * sequenceLength, dFF),
    
    gradient1(batch * sequenceLength, dFF),
    gradient2(batch * sequenceLength, dFF),
    gradient3(batch * sequenceLength, dFF) {;}

void FeedForwardBlock::forward(TensorView input, TensorView output) {
    linear1.forward(input, out1);
    relu.forward(out1, out2);
    dropout.forward(out2, out3);
    linear2.forward(out3, output);
}

void FeedForwardBlock::predict(TensorView input, TensorView output) {
    linear1.predict(input, out1);
    relu.predict(out1, out2);
    dropout.predict(out2, out3);
    linear2.predict(out3, output);
}

void FeedForwardBlock::backward(TensorView outputGradient, TensorView inputGradient, TensorView input) {
    linear2.backward(outputGradient, gradient3, out3);
    dropout.backward(gradient3, gradient2);
    relu.backward(gradient2, gradient1, out1);
    linear1.backward(gradient1, inputGradient, input);
}

void FeedForwardBlock::updateParameter() {
    linear1.updateParameter();
    linear2.updateParameter();
}
