#include "../Header.cuh"
#include "../Encoder.cuh"
#include "cnpy.h"

const std::string testCaseDir = "../../python/Testcase/Encoder";
const std::string modelName = "encoder";
const int feedTest = 5;
const int backTest = 5;

int main() {
    Tensor input(batch * sequenceLength, dModel);
	Tensor output(batch * sequenceLength, dModel);
	Tensor inGradient(batch * sequenceLength, dModel);
    Tensor outGradient(batch * sequenceLength, dModel);
    std::size_t* seq = new std::size_t(batch);
    Encoder model(input, output, inGradient, outGradient, seq);
    // param
    {
        cnpy::npz_t npFile = cnpy::npz_load(testCaseDir + "/" + modelName + "_param.npz");
        model.loadParam(npFile, "encoder");
    }
   
    // forwardTest
    for(int i = 0; i < feedTest;i++) {
        cnpy::npz_t npFile = cnpy::npz_load(testCaseDir + "/" + modelName + "_forward" + std::to_string(i) + ".npz");
        model.forwardTest(npFile, "encoder");
    }
    // backwardTest
    for(int i = 0; i < backTest;i++) {
        cnpy::npz_t npFile = cnpy::npz_load(testCaseDir + "/" + modelName + "_backward" + std::to_string(i) + ".npz");
        model.backwardTest(npFile, "encoder");
    }
    return 0;
}