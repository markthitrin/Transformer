#include "Config.h"
#include "Decoder.h"
#include "Embedding.h"
#include "Encoder.h"
#include "Header.h"
#include "LayerNorm.h"
#include "Linear.h"
#include "PositionalEncoder.h"
#include "Softmax.h"
#include "Tensor.h"
#include "Timer.h"
#include "Transformer.h"
#include "Util.h"

Transformer::Transformer() :
    srcEmbed(srcVocab),
    tgtEmbed(tgtVocab),
    linear(dModel, tgtVocab),

    encoderOut(batch * sequenceLength, dModel),
    encoderGradient(batch * sequenceLength, dModel),
    
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

void Transformer::forward(
    const int inpute[batch * sequenceLength], const int inputd[batch * sequenceLength], TensorView output,
    const int srcSeq[batch], const int tgtSeq[batch]) {

    srcEmbed.forward(inpute, out1);
    srcPos.forward(out1, out2);
    encoder.forward(out2, encoderOut, srcSeq);

    tgtEmbed.forward(inputd, out3);
    tgtPos.forward(out3, out4);
    decoder.forward(out4, encoderOut, out5, srcSeq, tgtSeq);
    linear.forward(out5, output);
}

void Transformer::predict(
    const int inpute[batch * sequenceLength], const int inputd[batch * sequenceLength], TensorView output,
    const int srcSeq[batch], const int tgtSeq[batch]) {

    srcEmbed.predict(inpute, out1);
    srcPos.predict(out1, out2);
    encoder.predict(out2, encoderOut, srcSeq);

    tgtEmbed.predict(inputd, out3);
    tgtPos.predict(out3, out4);
    decoder.predict(out4, encoderOut, out5, srcSeq, tgtSeq);
    linear.predict(out5, output);
}

void Transformer::backward(
    TensorView outputGradient,
    const int inpute[batch * sequenceLength], const int inputd[batch * sequenceLength],
    const int srcSeq[batch], const int tgtSeq[batch]) {

    linear.backward(outputGradient, gradient5, out5);
    decoder.backward(gradient5, gradient4, encoderGradient, encoderOut, srcSeq, tgtSeq);
    tgtPos.backward(gradient4, gradient3);
    tgtEmbed.backward(gradient3, inputd);

    encoder.backward(encoderGradient, gradient2, srcSeq);
    srcPos.backward(gradient2, gradient1);
    srcEmbed.backward(gradient1, inpute);
}

void Transformer::updateParameter() {
    srcEmbed.updateParameter();
    encoder.updateParameter();

    tgtEmbed.updateParameter();
    decoder.updateParameter();
    linear.updateParameter();
}