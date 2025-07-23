

#include "Header.h"
#include "Tensor.h"
#include "Util.h"
#include "cnpy.h"
#include "DropOut.h"
#include "PositionalEncoder.h"
#include "Timer.h"

PositionalEncoder::PositionalEncoder() :
    dropout(batch * sequenceLength, dModel),
    
    positionEncode(sequenceLength, dModel) {

    GetPositionalEncode(positionEncode);
}

void PositionalEncoder::forward(TensorView input, TensorView output) {
    #pragma omp parallel for schedule(static)
    for(int i = 0;i < batch;i++) {
        PlusPar(input.sliceRow(i * sequenceLength, sequenceLength), positionEncode, input.sliceRow(i * sequenceLength, sequenceLength));
    }
    Timer::CheckPoint();
    dropout.forward(input, output);
}

void PositionalEncoder::predict(TensorView input, TensorView output) {
    #pragma omp parallel for schedule(static)
    for(int i = 0;i < batch;i++) {
        PlusPar(input.sliceRow(i * sequenceLength, sequenceLength), positionEncode, output.sliceRow(i * sequenceLength, sequenceLength));
    }
}

void PositionalEncoder::backward(TensorView outputGradient, TensorView inputGradient) {
    dropout.backward(outputGradient, inputGradient);
    Timer::CheckPoint();
}

void PositionalEncoder::forwardTest(cnpy::npz_t npFile, std::string prefix) {
    Tensor target(batch * sequenceLength, dModel);
    Tensor input(batch * sequenceLength, dModel);
    Tensor output(batch * sequenceLength, dModel);

    input.loadNp(npFile, prefix + ".input");
    target.loadNp(npFile, prefix + ".output");

    forward(input, output);

    PrintTestResult("forward " + prefix, output, target);
}