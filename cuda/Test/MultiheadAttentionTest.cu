#include "../Header.cuh"
#include "../MultiheadAttention.cuh"
#include "cnpy.h"

const std::string testCaseDir = "../../python/Testcase/MultiheadAttention";
const std::string modelName = "multihead_attention";
const int feedTest = 5;
const int backTest = 5;

int main() {
    Tensor inputQ(batch * sequenceLength, dModel);
	Tensor inputK(batch * sequenceLength, dModel);
	Tensor inputV(batch * sequenceLength, dModel);
	Tensor output(batch * sequenceLength, dModel);
	Tensor inGradient(batch * sequenceLength, dModel);
	Tensor outGradientQ(batch * sequenceLength, dModel);
	Tensor outGradientK(batch * sequenceLength, dModel);
	Tensor outGradientV(batch * sequenceLength, dModel);
    std::size_t* seq = new std::size_t[batch];
    MultiheadAttention model(inputQ, inputK, inputV, output, inGradient, outGradientQ, outGradientK, outGradientV, LOOK_AHEAD, seq);
    // param
    {
        cnpy::npz_t npFile = cnpy::npz_load(testCaseDir + "/" + modelName + "_param.npz");
        model.loadParam(npFile, "multiheadAtt");
    }
   
    // forwardTest
    for(int i = 0; i < feedTest;i++) {
        cnpy::npz_t npFile = cnpy::npz_load(testCaseDir + "/" + modelName + "_forward" + std::to_string(i) + ".npz");
        model.forwardTest(npFile, "multiheadAtt");
    }
    // backwardTest
    for(int i = 0; i < backTest;i++) {
        cnpy::npz_t npFile = cnpy::npz_load(testCaseDir + "/" + modelName + "_backward" + std::to_string(i) + ".npz");
        model.backwardTest(npFile, "multiheadAtt");
    }
    return 0;
}