#include "../Header.h"
#include "../DecoderLayer.h"
#include "cnpy.h"

const std::string testCaseDir = "../../python/Testcase/DecoderBlock";
const std::string modelName = "decoder_block";
const int feedTest = 5;
const int backTest = 5;

int main() {
    Tensor<batch * sequenceLength, dModel> input;
	Tensor<batch * sequenceLength, dModel> output;
    Tensor<batch * sequenceLength, dModel> encoderOut;
	Tensor<batch * sequenceLength, dModel> inGradient;
	Tensor<batch * sequenceLength, dModel> outGradient;
    Tensor<batch * sequenceLength, dModel> encoderGradient;
    DecoderLayer<batch, sequenceLength, dModel> model(input, encoderOut, output, inGradient, outGradient, encoderGradient);
    // param
    {
        cnpy::npz_t npFile = cnpy::npz_load(testCaseDir + "/" + modelName + "_param.npz");
        model.loadParam(npFile, "decoderBlock");
    }
   
    // forwardTest
    for(int i = 0; i < feedTest;i++) {
        cnpy::npz_t npFile = cnpy::npz_load(testCaseDir + "/" + modelName + "_forward" + std::to_string(i) + ".npz");
        model.forwardTest(npFile, "decoderBlock");
    }
    // backwardTest
    for(int i = 0; i < backTest;i++) {
        cnpy::npz_t npFile = cnpy::npz_load(testCaseDir + "/" + modelName + "_backward" + std::to_string(i) + ".npz");
        model.backwardTest(npFile, "decoderBlock");
    }
    return 0;
}