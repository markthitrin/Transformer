#include "../Header.h"
#include "../EncoderLayer.h"
#include "cnpy.h"

const std::string testCaseDir = "../../python/Testcase/EncoderBlock";
const std::string modelName = "encoder_block";
const int feedTest = 5;
const int backTest = 5;

int main() {
    EncoderLayer model;
    // param
    {
        cnpy::npz_t npFile = cnpy::npz_load(testCaseDir + "/" + modelName + "_param.npz");
        model.loadParam(npFile, "encoderBlock");
    }
   
    // forwardTest
    for(int i = 0; i < feedTest;i++) {
        cnpy::npz_t npFile = cnpy::npz_load(testCaseDir + "/" + modelName + "_forward" + std::to_string(i) + ".npz");
        model.forwardTest(npFile, "encoderBlock");
    }
    // backwardTest
    for(int i = 0; i < backTest;i++) {
        cnpy::npz_t npFile = cnpy::npz_load(testCaseDir + "/" + modelName + "_backward" + std::to_string(i) + ".npz");
        model.backwardTest(npFile, "encoderBlock");
    }
    return 0;
}