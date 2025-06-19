#include "../Header.h"
#include "../Transformer.h"
#include "cnpy.h"

const std::string testCaseDir = "../../python/Testcase/Transformer";
const std::string modelName = "transformer";
const int feedTest = 5;
const int backTest = 5;

int main() {
    Tensor<1, batch * sequenceLength> inputEncoder;
    Tensor<1, batch * sequenceLength> inputDecoder;
	Tensor<batch * sequenceLength, tgtVocab> output;
	Tensor<batch * sequenceLength, tgtVocab> inGradient;
    Transformer model(inputEncoder, inputDecoder, output, inGradient);
    // param
    {
        cnpy::npz_t npFile = cnpy::npz_load(testCaseDir + "/" + modelName + "_param.npz");
        model.loadParam(npFile, "transformer");
    }
   
    // forwardTest
    for(int i = 0; i < feedTest;i++) {
        cnpy::npz_t npFile = cnpy::npz_load(testCaseDir + "/" + modelName + "_forward" + std::to_string(i) + ".npz");
        model.forwardTest(npFile, "transformer");
    }
    // backwardTest
    for(int i = 0; i < backTest;i++) {
        cnpy::npz_t npFile = cnpy::npz_load(testCaseDir + "/" + modelName + "_backward" + std::to_string(i) + ".npz");
        model.backwardTest(npFile, "transformer");
    }
    return 0;
}