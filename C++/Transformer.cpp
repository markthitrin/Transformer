#include "Header.h"
#include "Tensor.h"
#include "Encoder.h"
#include "Softmax.h"
#include "PositionalEncoder.h"
#include "Embedding.h"
#include "Linear.h"
#include "LayerNorm.h"
#include "Util.h"
#include "Decoder.h"
#include "Transformer.h"

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

void Transformer::loadParam(cnpy::npz_t npFile, std::string prefix) {
    encoder.loadParam(npFile, prefix + ".encoder");
    decoder.loadParam(npFile, prefix + ".decoder");
    srcEmbed.loadParam(npFile, prefix + ".src_embed");
    tgtEmbed.loadParam(npFile, prefix + ".tgt_embed");
    linear.weight.loadNp(npFile, prefix + ".projection_layer.weight");
    linear.bias.loadNp(npFile, prefix + ".projection_layer.bias");
}

void Transformer::checkUpdatedParam(cnpy::npz_t npFile, std::string prefix) {
    encoder.checkUpdatedParam(npFile, prefix + ".encoder");
    decoder.checkUpdatedParam(npFile, prefix + ".decoder");
    srcEmbed.checkUpdatedParam(npFile, prefix + ".src_embed");
    tgtEmbed.checkUpdatedParam(npFile, prefix + ".tgt_embed");
    linear.checkUpdatedParam(npFile, prefix + ".projection_layer");
}

void Transformer::forwardTest(cnpy::npz_t npFile, std::string prefix) {
    Tensor target(batch * sequenceLength, tgtVocab);
    Tensor inputEncoder(1, batch * sequenceLength);
    Tensor inputDecoder(1, batch * sequenceLength);
    Tensor output(batch * sequenceLength, tgtVocab);
    Tensor npdLoad(1,2);
    int srcSeq[batch];
    int tgtSeq[batch];
    int inpute[batch* sequenceLength];
    int inputd[batch* sequenceLength];

    inputEncoder.loadNp(npFile, prefix + ".input1");
    inputDecoder.loadNp(npFile, prefix + ".input2");
    target.loadNp(npFile, prefix + ".output");
    npdLoad.loadNp(npFile, prefix + ".npd");
    for(int i = 0;i < batch;i++) srcSeq[i] = npdLoad[0];
    for(int i = 0;i < batch;i++) tgtSeq[i] = npdLoad[1];
    for(int i = 0;i < batch * sequenceLength;i++) inpute[i] = inputEncoder[i];
    for(int i = 0;i < batch * sequenceLength;i++) inputd[i] = inputDecoder[i];

    forward(inpute, inputd, output, srcSeq, tgtSeq);

    PrintTestResult("forward", output, target);
}

void Transformer::backwardTest(cnpy::npz_t npFile, std::string prefix) {
    Tensor target(batch * sequenceLength, tgtVocab);
    Tensor inputEncoder(1, batch * sequenceLength);
    Tensor inputDecoder(1, batch * sequenceLength);
    Tensor output(batch * sequenceLength, tgtVocab);
    Tensor outputGradient(batch * sequenceLength, tgtVocab);
    Tensor npdLoad(1,2);
    int srcSeq[batch];
    int tgtSeq[batch];
    int inpute[batch* sequenceLength];
    int inputd[batch* sequenceLength];

    outputGradient = 1.0f / outputGradient.row/ outputGradient.col;

    inputEncoder.loadNp(npFile, prefix + ".input1");
    inputDecoder.loadNp(npFile, prefix + ".input2");
    npdLoad.loadNp(npFile, prefix + ".npd");
    for(int i = 0;i < batch;i++) srcSeq[i] = npdLoad[0];
    for(int i = 0;i < batch;i++) tgtSeq[i] = npdLoad[1];
    for(int i = 0;i < batch * sequenceLength;i++) inpute[i] = inputEncoder[i];
    for(int i = 0;i < batch * sequenceLength;i++) inputd[i] = inputDecoder[i];

    forward(inpute, inputd, output, srcSeq, tgtSeq);
    backward(outputGradient, inpute, inputd, srcSeq, tgtSeq);
    updateParameter();

    checkUpdatedParam(npFile, prefix);
}