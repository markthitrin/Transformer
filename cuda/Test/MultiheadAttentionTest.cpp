#include "../Header.h"
#include "../MultiheadAttention.h"
#include "cnpy.h"

const std::string testCaseDir = "../../python/Testcase/MultiheadAttention";
const std::string modelName = "multihead_attention";
const int feedTest = 5;
const int backTest = 5;

int main() {
    Tensor<batch * sequenceLength, dModel> inputQ;
	Tensor<batch * sequenceLength, dModel> inputK;
	Tensor<batch * sequenceLength, dModel> inputV;
	Tensor<batch * sequenceLength, dModel> output;
	Tensor<batch * sequenceLength, dModel> inGradient;
	Tensor<batch * sequenceLength, dModel> outGradientQ;
	Tensor<batch * sequenceLength, dModel> outGradientK;
	Tensor<batch * sequenceLength, dModel> outGradientV;
    MultiheadAttention<head, 0, batch,  sequenceLength, dModel> model(inputQ, inputK, inputV, output, inGradient, outGradientQ, outGradientK, outGradientV);
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