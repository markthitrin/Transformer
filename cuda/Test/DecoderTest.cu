#include "../Header.cuh"
#include "../Decoder.cuh"
#include "cnpy.h"

const std::string testCaseDir = "../../python/Testcase/Decoder";
const std::string modelName = "decoder";
const int feedTest = 5;
const int backTest = 5;

int main() {
    Tensor input(batch * sequenceLength, dModel);
	Tensor output(batch * sequenceLength, dModel);
    Tensor encoderOut(batch * sequenceLength, dModel);
	Tensor inGradient(batch * sequenceLength, dModel);
	Tensor outGradient(batch * sequenceLength, dModel);
    Tensor encoderGradient(batch * sequenceLength, dModel);
    std::size_t* srcSeqH = new std::size_t[batch];
    std::size_t* tgtSeqH = new std::size_t[batch];
    Decoder model(input, encoderOut, output, inGradient, outGradient, encoderGradient, srcSeqH, tgtSeqH);
    // param
    {
        cnpy::npz_t npFile = cnpy::npz_load(testCaseDir + "/" + modelName + "_param.npz");
        model.loadParam(npFile, "decoder");
    }
   
    // forwardTest
    // for(int i = 0; i < feedTest;i++) {
    //     cnpy::npz_t npFile = cnpy::npz_load(testCaseDir + "/" + modelName + "_forward" + std::to_string(i) + ".npz");
    //     model.forwardTest(npFile, "decoder");
    // }
    // backwardTest
    for(int i = 0; i < backTest;i++) {
        cnpy::npz_t npFile = cnpy::npz_load(testCaseDir + "/" + modelName + "_backward" + std::to_string(i) + ".npz");
        model.backwardTest(npFile, "decoder");
    }
    return 0;
}