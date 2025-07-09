#ifndef DATA
#define DATA

#include "Header.h"
#include "Tensor.h"

class Data {
public:
    Data(std::vector<std::vector<int>>&& srcTokens, std::vector<std::vector<int>>&& tgtToken);

    void get(int* inpute, int* inputd, int* target, int* seqSrq, int* tgtSeq);

    std::vector<std::vector<int>> srcTokens;
    std::vector<std::vector<int>> tgtToken;
};

#endif