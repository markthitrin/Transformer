#include "Config.h"
#include "DropOut.h"
#include "EncoderLayer.h"
#include "Header.h"
#include "FeedForwardBlock.h"
#include "LayerNorm.h"
#include "MultiheadAttention.h"
#include "Tensor.h"
#include "Util.h"

EncoderLayer::EncoderLayer() :
    dropout1(batch * sequenceLength, dModel),
    dropout2(batch * sequenceLength, dModel),

    out1(batch * sequenceLength, dModel),
    out2(batch * sequenceLength, dModel),
    out3(batch * sequenceLength, dModel),
    out4(batch * sequenceLength, dModel),
    out5(batch * sequenceLength, dModel),

    gradient1(batch * sequenceLength, dModel),
    gradient2(batch * sequenceLength, dModel),
    gradient3(batch * sequenceLength, dModel),
    gradient4(batch * sequenceLength, dModel),
    gradient5(batch * sequenceLength, dModel) {;}

void EncoderLayer::forward(TensorView input, TensorView output, const int srcSeq[batch]) {
    norm1.forward(input, out1);
    mulAtt.forward(out1, out1, out1, out2, PADDING, srcSeq);
    dropout1.forward(out2, out3);
    Plus(input, out3, out3);

    norm2.forward(out3, out4);
    pff.forward(out4, out5);
    dropout2.forward(out5, output);
    Plus(out3, output, output);
}

void EncoderLayer::predict(TensorView input, TensorView output, const int srcSeq[batch]) {
    norm1.predict(input, out1);
    mulAtt.predict(out1, out1, out1, out2, PADDING, srcSeq);
    dropout1.predict(out2, out3);
    Plus(input, out3, out3);

    norm2.predict(out3, out4);
    pff.predict(out4, out5);
    dropout2.predict(out5, output);
    Plus(out3, output, output);
}

void EncoderLayer::backward(TensorView outputGradient, TensorView inputGradient, const int srcSeq[batch]) {
    dropout2.backward(outputGradient, gradient5);
    pff.backward(gradient5, gradient4, out4);
    norm2.backward(gradient4, gradient3);
    Plus(outputGradient, gradient3, gradient3);

    dropout1.backward(gradient3, gradient2);
    mulAtt.backward(gradient2, gradient1, gradient1, gradient1, out1, out1, out1, out2, PADDING, srcSeq);
    norm1.backward(gradient1, inputGradient);
    Plus(gradient3, inputGradient, inputGradient);
}

void EncoderLayer::updateParameterTask() {
    norm1.updateParameterTask();
    mulAtt.updateParameterTask();
    norm2.updateParameterTask();
    pff.updateParameterTask();
}