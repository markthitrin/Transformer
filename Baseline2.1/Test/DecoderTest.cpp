#include "../Header.h"
#include "../Decoder.h"
#include "cnpy.h"

const std::string testCaseDir = "../../python/Testcase/Decoder";
const std::string modelName = "decoder";
const int feedTest = 5;
const int backTest = 5;

int main() {
    Decoder<batch, sequenceLength, dModel, 6> model;
    // param
    {
        cnpy::npz_t npFile = cnpy::npz_load(testCaseDir + "/" + modelName + "_param.npz");
        model.loadParam(npFile, "decoder");
    }
   
    // forwardTest
    for(int i = 0; i < feedTest;i++) {
        cnpy::npz_t npFile = cnpy::npz_load(testCaseDir + "/" + modelName + "_forward" + std::to_string(i) + ".npz");
        model.forwardTest(npFile, "decoder");
    }
    // backwardTest
    for(int i = 0; i < backTest;i++) {
        cnpy::npz_t npFile = cnpy::npz_load(testCaseDir + "/" + modelName + "_backward" + std::to_string(i) + ".npz");
        model.backwardTest(npFile, "decoder");
    }
    return 0;
}