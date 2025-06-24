#include "../Header.cuh"
#include "../PositionalEncoder.cuh"
#include "cnpy.h"

const std::string testCaseDir = "../../python/Testcase/PositionalEncoding";
const std::string modelName = "positional_encoding";
const int feedTest = 5;
const int backTest = 5;

int main() {
    Tensor input(batch * sequenceLength, dModel);
    Tensor output(batch * sequenceLength, dModel);
    Tensor inGradient(batch * sequenceLength, dModel);
    Tensor outGradient(batch * sequenceLength, dModel);
    PositionalEncoder model(input, output, inGradient, outGradient);
    // param
    // forwardTest
    for(int i = 0; i < feedTest;i++) {
        cnpy::npz_t npFile = cnpy::npz_load(testCaseDir + "/" + modelName + "_forward" + std::to_string(i) + ".npz");
        model.forwardTest(npFile, "positionalEncoding");
    }
    // backwardTest
    return 0;
}