#include "../Header.h"
#include "../PositionalEncoder.h"
#include "cnpy.h"

const std::string testCaseDir = "../../python/Testcase/PositionalEncoding";
const std::string modelName = "positional_encoding";
const int feedTest = 5;
const int backTest = 5;

int main() {
    PositionalEncoder model;
    // param
    // forwardTest
    for(int i = 0; i < feedTest;i++) {
        cnpy::npz_t npFile = cnpy::npz_load(testCaseDir + "/" + modelName + "_forward" + std::to_string(i) + ".npz");
        model.forwardTest(npFile, "positionalEncoding");
    }
    // backwardTest
    return 0;
}