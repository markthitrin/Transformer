#include "../Header.cuh"
#include "../DecoderLayer.cuh"
#include "cnpy.h"

const std::string testCaseDir = "../../python/Testcase/DecoderBlock";
const std::string modelName = "decoder_block";
const int feedTest = 5;
const int backTest = 5;

int main() {
    Tensor input(batch * sequenceLength, dModel);
	Tensor output(batch * sequenceLength, dModel);
    Tensor encoderOut(batch * sequenceLength, dModel);
	Tensor inGradient(batch * sequenceLength, dModel);
	Tensor outGradient(batch * sequenceLength, dModel);
    Tensor encoderGradient(batch * sequenceLength, dModel);
    std::size_t* seq1 = new std::size_t[batch];
    std::size_t* seq2 = new std::size_t[batch];
    DecoderLayer model(input, encoderOut, output, inGradient, outGradient, encoderGradient, seq1, seq2);
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