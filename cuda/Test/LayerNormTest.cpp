#include "../Header.h"
#include "../LayerNorm.h"
#include "cnpy.h"

const std::string testCaseDir = "../../python/Testcase/LayerNorm";
const std::string modelName = "layer_norm";
const int feedTest = 5;
const int backTest = 5;

int main() {
    Tensor<batch * sequenceLength, dModel> input;
	Tensor<batch * sequenceLength, dModel> output;
	Tensor<batch * sequenceLength, dModel> inGradient;
	Tensor<batch * sequenceLength, dModel> outGradient;
    LayerNorm<batch * sequenceLength, dModel> model(input, output, inGradient, outGradient);
    // param
    {
        cnpy::npz_t npFile = cnpy::npz_load(testCaseDir + "/" + modelName + "_param.npz");
        model.loadParam(npFile, "layerNorm");
    }
   
    // forwardTest
    for(int i = 0; i < feedTest;i++) {
        cnpy::npz_t npFile = cnpy::npz_load(testCaseDir + "/" + modelName + "_forward" + std::to_string(i) + ".npz");
        model.forwardTest(npFile, "layerNorm");
    }
    // backwardTest
    for(int i = 0; i < backTest;i++) {
        cnpy::npz_t npFile = cnpy::npz_load(testCaseDir + "/" + modelName + "_backward" + std::to_string(i) + ".npz");
        model.backwardTest(npFile, "layerNorm");
    }
    return 0;
}