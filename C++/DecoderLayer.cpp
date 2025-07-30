#include "Config.h"
#include "DecoderLayer.h"
#include "DropOut.h"
#include "FeedForwardBlock.h"
#include "Header.h"
#include "LayerNorm.h"
#include "MultiheadAttention.h"
#include "Tensor.h"

DecoderLayer::DecoderLayer():
    dropout1(batch * sequenceLength, dModel),
    dropout2(batch * sequenceLength, dModel),
    dropout3(batch * sequenceLength, dModel),

    out1(batch * sequenceLength, dModel),
    out2(batch * sequenceLength, dModel),
    out3(batch * sequenceLength, dModel),
    out4(batch * sequenceLength, dModel),
    out5(batch * sequenceLength, dModel),
    out6(batch * sequenceLength, dModel),
    out7(batch * sequenceLength, dModel),
    out8(batch * sequenceLength, dModel),

    gradient1(batch * sequenceLength, dModel),
    gradient2(batch * sequenceLength, dModel),
    gradient3(batch * sequenceLength, dModel),
    gradient4(batch * sequenceLength, dModel),
    gradient5(batch * sequenceLength, dModel),
    gradient6(batch * sequenceLength, dModel),
    gradient7(batch * sequenceLength, dModel),
    gradient8(batch * sequenceLength, dModel) {;}

void DecoderLayer::forward(
    TensorView input, TensorView encoderOutput, TensorView output, 
    const int srcSeq[batch], const int tgtSeq[batch]) {

    norm1.forward(input, out1);
    mulAtt1.forward(out1, out1, out1, out2, LOOK_AHEAD, tgtSeq);
    dropout1.forward(out2, out3);
    Plus(input, out3, out3);

    norm2.forward(out3, out4);
    mulAtt2.forward(out4, encoderOutput, encoderOutput, out5, CROSS_PADDING, srcSeq);
    dropout2.forward(out5, out6);
    Plus(out3, out6, out6);

    norm3.forward(out6, out7);
    pff.forward(out7, out8);
    dropout3.forward(out8, output);
    Plus(out6, output, output);
}

void DecoderLayer::predict(
    TensorView input, TensorView encoderOutput, TensorView output, 
    const int srcSeq[batch], const int tgtSeq[batch]) {

    norm1.predict(input, out1);
    mulAtt1.predict(out1, out1, out1, out2, LOOK_AHEAD, tgtSeq);
    dropout1.predict(out2, out3);
    Plus(input, out3, out3);

    norm2.predict(out3, out4);
    mulAtt2.predict(out4, encoderOutput, encoderOutput, out5, CROSS_PADDING, srcSeq);
    dropout2.predict(out5, out6);
    Plus(out3, out6, out6);

    norm3.predict(out6, out7);
    pff.predict(out7, out8);
    dropout3.predict(out8, output);
    Plus(out6, output, output);
}

void DecoderLayer::backward(
    TensorView outputGradient, TensorView encoderGradient, TensorView inputGradient,
    TensorView encoderOut, const int srcSeq[batch], const int tgtSeq[batch]) {

    dropout3.backward(outputGradient, gradient8);
    pff.backward(gradient8, gradient7, out7);
    norm3.backward(gradient7, gradient6);
    Plus(outputGradient, gradient6, gradient6);

    dropout2.backward(gradient6, gradient5);
    mulAtt2.backward(gradient5, gradient4, encoderGradient, encoderGradient, out4, encoderOut, encoderOut, out5, CROSS_PADDING, srcSeq);
    norm2.backward(gradient4, gradient3);
    Plus(gradient6, gradient3, gradient3);

    dropout1.backward(gradient3, gradient2);
    mulAtt1.backward(gradient2, gradient1, gradient1, gradient1, out1, out1, out1, out2, LOOK_AHEAD, tgtSeq);
    norm1.backward(gradient1, inputGradient);
    Plus(gradient3, inputGradient, inputGradient);
}

void DecoderLayer::updateParameterTask() {
    norm1.updateParameterTask();
    mulAtt1.updateParameterTask();
    norm2.updateParameterTask();
    mulAtt2.updateParameterTask();
    norm3.updateParameterTask();
    pff.updateParameterTask();
}

void DecoderLayer::loadParam(cnpy::npz_t npFile, std::string prefix) {
    norm1.loadParam(npFile, prefix + ".sub1.layerNorm");
    mulAtt1.loadParam(npFile, prefix + ".sub1.sublayer");
    norm2.loadParam(npFile, prefix + ".sub2.layerNorm");
    mulAtt2.loadParam(npFile, prefix + ".sub2.sublayer");
    norm3.loadParam(npFile, prefix + ".sub3.layerNorm");
    pff.loadParam(npFile,prefix + ".sub3.sublayer");
}

void DecoderLayer::checkUpdatedParam(cnpy::npz_t npFile, std::string prefix) {
    norm1.checkUpdatedParam(npFile, prefix + ".sub1.layerNorm");
    mulAtt1.checkUpdatedParam(npFile, prefix + ".sub1.sublayer");
    norm2.checkUpdatedParam(npFile, prefix + ".sub2.layerNorm");
    mulAtt2.checkUpdatedParam(npFile, prefix + ".sub2.sublayer");
    norm3.checkUpdatedParam(npFile, prefix + ".sub3.layerNorm");
    pff.checkUpdatedParam(npFile, prefix + ".sub3.sublayer");
}

void DecoderLayer::forwardTest(cnpy::npz_t npFile, std::string prefix) {
    Tensor target(batch * sequenceLength, dModel);
    Tensor target1(batch * sequenceLength, dModel);
    Tensor target2(batch * sequenceLength, dModel);
    Tensor input1(batch * sequenceLength, dModel);
    Tensor input2(batch * sequenceLength, dModel);
    Tensor output(batch * sequenceLength, dModel);
    Tensor npdLoad(1,2);
    int srcSeq[batch];
    int tgtSeq[batch];

    input1.loadNp(npFile, prefix + ".input1");
    input2.loadNp(npFile, prefix + ".input2");
    target.loadNp(npFile, prefix + ".output");
    // target1.loadNp(npFile, prefix + ".output1");
    // target2.loadNp(npFile, prefix + ".output2");
    npdLoad.loadNp(npFile, prefix + ".npd");
    for(int i = 0;i < batch;i++) srcSeq[i] = npdLoad[0];
    for(int i = 0;i < batch;i++) tgtSeq[i] = npdLoad[1];

    forward(input1, input2, output, srcSeq, tgtSeq);

    PrintTestResult("forward", output, target);
    // PrintTestResult("forward", out3, target1);
    // PrintTestResult("forward", out6, target2);
}

void DecoderLayer::backwardTest(cnpy::npz_t npFile, std::string prefix) {
    Tensor target(batch * sequenceLength, dModel);
    Tensor input1(batch * sequenceLength, dModel);
    Tensor input2(batch * sequenceLength, dModel);
    Tensor output(batch * sequenceLength, dModel);
    Tensor outputGradient(batch * sequenceLength, dModel);
    Tensor inputGradient(batch * sequenceLength, dModel);
    Tensor npdLoad(1,2);
    int srcSeq[batch];
    int tgtSeq[batch];

    outputGradient = 1.0f / outputGradient.row / outputGradient.col;

    input1.loadNp(npFile, prefix + ".input1");
    input2.loadNp(npFile, prefix + ".input2");
    npdLoad.loadNp(npFile, prefix + ".npd");
    for(int i = 0;i < batch;i++) srcSeq[i] = npdLoad[0];
    for(int i = 0;i < batch;i++) tgtSeq[i] = npdLoad[1];

    forward(input1, input2, output, srcSeq, tgtSeq);
    backward(outputGradient, inputGradient, inputGradient, input2, srcSeq, tgtSeq);
    #pragma omp parallel
    {
        #pragma omp single
        {
            updateParameterTask();
            #pragma omp taskwait
        }
    }

    checkUpdatedParam(npFile, prefix);
}