#include <iostream>
#include "Tensor.h"


template<int batch, int len>
class Data {
public:
    Data(std::vector<std::vector<float>>&& srcTokens, std::vector<std::vector<float>>&& tgtToken) : srcTokens(srcTokens), tgtToken(tgtToken) {

    }
    void get(Tensor<1, batch * len> source, Tensor<1, batch * len> tragetInput, Tensor<1, batch * len> targetOutput,int npdSrc[batch],int npdTgt[batch]) {
        for(int i = 0;i < batch;i++) {
            int pos = std::rand() % srcTokens.size();
            int srclen = srcTokens[pos].size();
            if(srclen > sequenceLength - 1) {
                --i;
                continue;
            }
            int tgtlen = std::rand() % (tgtToken[pos].size() + 1);
            float* rowSrc = source.data + i * len;
            float* rowTgtI = tragetInput.data + i * len;
            float* rowTgtO = targetOutput.data + i * len;
            for(int j = 0;j < srclen;j++) {
                rowSrc[j] = srcTokens[pos][j];
            }
            for(int j = srclen;j < len;j++) {
                rowSrc[j] = 1.0f; // padding
            }

            rowTgtI[0] = 2.0f; // sos
            for(int j = 1;j < tgtlen + 1;j++) {
                rowTgtI[j] = tgtToken[pos][j - 1];
            }
            for(int j = tgtlen + 1;j < len;j++) {
                rowTgtI[j] = 1.0f; // padding
            }

            for(int j = 0;j < tgtlen;j++) {
                rowTgtO[j] = tgtToken[pos][j];
            }
            if(tgtToken[pos].size() <= tgtlen) {
                rowTgtO[tgtlen] = 3.0f; // eos
            }
            else {
                rowTgtO[tgtlen] = tgtToken[pos][tgtlen];
            }
            for(int j = tgtlen + 1;j < len;j++) {
                rowTgtO[j] = 1.0f; // padding
            }

            npdSrc[i] = srclen;
            npdTgt[i] = tgtlen;
        }
    }
    std::vector<std::vector<float>> srcTokens;
    std::vector<std::vector<float>> tgtToken;
};