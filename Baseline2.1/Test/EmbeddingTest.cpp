#include "../Header.h"
#include "../Embedding.h"
#include "cnpy.h"

const std::string testCaseDir = "../../python/Testcase/InputEmbeddings";
const std::string modelName = "embeddings";
const int feedTest = 5;
const int backTest = 5;

int main() {
    Embedding<batch * sequenceLength, 6, dModel> model;
    // param
    {
        cnpy::npz_t npFile = cnpy::npz_load(testCaseDir + "/" + modelName + "_param.npz");
        model.loadParam(npFile, "embeddings");
    }
   
    // forwardTest
    for(int i = 0; i < feedTest;i++) {
        cnpy::npz_t npFile = cnpy::npz_load(testCaseDir + "/" + modelName + "_forward" + std::to_string(i) + ".npz");
        model.forwardTest(npFile, "embeddings");
    }
    // backwardTest
    for(int i = 0; i < backTest;i++) {
        cnpy::npz_t npFile = cnpy::npz_load(testCaseDir + "/" + modelName + "_backward" + std::to_string(i) + ".npz");
        model.backwardTest(npFile, "embeddings");
    }
    return 0;
}