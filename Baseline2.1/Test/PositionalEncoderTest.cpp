#include "../Header.h"
#include "../PositionalEncoder.h"
#include "cnpy.h"

const std::string testCaseDir = "../../python/Testcase/PositionalEncoding";
const std::string modelName = "positional_encoding";
const int feedTest = 5;
const int backTest = 5;

int main() {
    Tensor<batch * sequenceLength, dModel> input;
    Tensor<batch * sequenceLength, dModel> output;
    Tensor<batch * sequenceLength, dModel> inGradient;
    Tensor<batch * sequenceLength, dModel> outGradient;
    PositionalEncoder<batch, sequenceLength, dModel> model(input, output, inGradient, outGradient);
    // param
    // forwardTest
    for(int i = 0; i < feedTest;i++) {
        cnpy::npz_t npFile = cnpy::npz_load(testCaseDir + "/" + modelName + "_forward" + std::to_string(i) + ".npz");
        model.forwardTest(npFile, "positionalEncoding");
    }
    // backwardTest
    return 0;
}