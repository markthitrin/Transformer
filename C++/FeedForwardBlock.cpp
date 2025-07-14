#include "Header.h"
#include "Tensor.h"
#include "Linear.h"
#include "ReLU.h"
#include "Util.h"
#include "LayerNorm.h"
#include "DropOut.h"
#include "FeedForwardBlock.h"

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

void FeedForwardBlock::loadParam(cnpy::npz_t npFile, std::string prefix) {
    linear1.weight.loadNp(npFile, prefix + ".w1");
    linear1.bias.loadNp(npFile, prefix + ".b1");
    linear2.weight.loadNp(npFile, prefix + ".w2");
    linear2.bias.loadNp(npFile, prefix + ".b2");
}

void FeedForwardBlock::checkUpdatedParam(cnpy::npz_t npFile, std::string prefix) {
    Tensor w1Updated(dModel, dFF);
    Tensor b1Updated(1, dFF);
    Tensor w2Updated(dFF, dModel);
    Tensor b2Updated(1, dModel);
    w1Updated.loadNp(npFile, prefix + ".updated_w1");
    b1Updated.loadNp(npFile, prefix + ".updated_b1");
    w2Updated.loadNp(npFile, prefix + ".updated_w2");
    b2Updated.loadNp(npFile, prefix + ".updated_b2");

    PrintTestResult("backward " + prefix + ".w1", linear1.weight, w1Updated);
    PrintTestResult("backward " + prefix + ".b1", linear1.bias, b1Updated);
    PrintTestResult("backward " + prefix + ".w2", linear2.weight, w2Updated);
    PrintTestResult("backward " + prefix + ".b2", linear2.bias, b2Updated);
}

void FeedForwardBlock::forwardTest(cnpy::npz_t npFile, std::string prefix) {
    Tensor input(batch * sequenceLength, dModel);
    Tensor output(batch * sequenceLength, dModel);
    Tensor target(batch * sequenceLength, dModel);

    input.loadNp(npFile, prefix + ".input");
    target.loadNp(npFile, prefix + ".output");

    forward(input, output);
    PrintTestResult("forward", output, target);
}

void FeedForwardBlock::backwardTest(cnpy::npz_t npFile, std::string prefix) {
    Tensor input(batch * sequenceLength, dModel);
    Tensor output(batch * sequenceLength, dModel);
    Tensor outputGradient(batch * sequenceLength, dModel);
    Tensor inputGradient(batch * sequenceLength, dModel);
    outputGradient = 1.0f / outputGradient.row / outputGradient.col;

    input.loadNp(npFile, prefix + ".input");
    
    forward(input, output);
    backward(outputGradient, inputGradient, input);
    updateParameter();

    checkUpdatedParam(npFile, prefix);
}
