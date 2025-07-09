#include "Header.h"
#include "Tensor.h"
#include "Data.h"

Data::Data(std::vector<std::vector<int>>&& srcTokens, std::vector<std::vector<int>>&& tgtToken) 
    : srcTokens(srcTokens), tgtToken(tgtToken) {}

void Data::get(int* inpute, int* inputd, int* target, int* seqSrq, int* tgtSeq) {
    for(int i = 0;i < batch;i++) {
        int pos = std::rand() % srcTokens.size();
        int srclen = srcTokens[pos].size();
        int tgtlen = std::rand() % (tgtToken[pos].size() + 1);
        if(srclen > sequenceLength - 1 || tgtlen + 1 > sequenceLength) {
            --i;
            continue;
        }
        

        for(int j = 0;j < srclen;j++) {
            inpute[j] = srcTokens[pos][j];
        }
        for(int j = srclen;j < sequenceLength;j++) {
            inpute[j] = 1; // padding
        }

        inputd[0] = 2; // sos
        for(int j = 1;j < tgtlen + 1;j++) {
            inputd[j] = tgtToken[pos][j - 1];
        }
        for(int j = tgtlen + 1;j < sequenceLength;j++) {
            inputd[j] = 1; // padding
        }

        for(int j = 0;j < tgtlen;j++) {
            target[j] = tgtToken[pos][j];
        }
        if(tgtToken[pos].size() == tgtlen) {
            target[tgtlen] = 3; // eos
        }
        else {
            target[tgtlen] = tgtToken[pos][tgtlen];
        }
        for(int j = tgtlen + 1;j < sequenceLength;j++) {
            target[j] = 1; // padding
        }

        seqSrq[i] = srclen;
        tgtSeq[i] = tgtlen + 1;

        inpute += sequenceLength;
        inputd += sequenceLength;
        target += sequenceLength;
    }
}
