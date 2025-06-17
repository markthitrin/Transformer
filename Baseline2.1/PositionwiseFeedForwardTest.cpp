#include "Header.h"
#include "PositionwiseFeedForward.h"
#include "cnpy.h"

const std::string testCaseDir = "../python/Testcase/FeedForwardBlock";
const std::string modelName = "feedforward";
const int feedTest = 5;
const int backTest = 5;

int main() {
    PositionwiseFeedForward<batch * sequenceLength, dModel, dFF> model;
    // param
    {
        cnpy::npz_t npFile = cnpy::npz_load(testCaseDir + "/" + modelName + "_param.npz");
        model.loadParam(npFile, "feedforward");
    }
   
    // forwardTest
    // for(int i = 0; i < feedTest;i++) {
    //     cnpy::npz_t npFile = cnpy::npz_load(testCaseDir + "/" + modelName + "_forward" + std::to_string(i) + ".npz");
    //     model.forwardTest(npFile, "feedforward");
    // }
    // backwardTest
    for(int i = 0; i < backTest;i++) {
        cnpy::npz_t npFile = cnpy::npz_load(testCaseDir + "/" + modelName + "_backward" + std::to_string(i) + ".npz");
        model.backwardTest(npFile, "feedforward");
    }
    return 0;
}