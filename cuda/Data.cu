#include "Header.cuh"
#include "Config.h"
#include "Data.cuh"

Data::Data(std::vector<std::vector<std::size_t>>&& srcTokens, std::vector<std::vector<std::size_t>>&& tgtToken) noexcept :
    srcTokens(srcTokens), tgtToken(tgtToken) {;}

void Data::get(std::size_t* encoderInput, std::size_t* srcSeq,
    std::size_t* decoderInput, std::size_t* tgtSeq,
    std::size_t* tragetOutput) {
    
    for(int i = 0;i < batch;i++) {
        std::size_t pos = std::rand() % srcTokens.size();
        std::size_t srclen = srcTokens[pos].size();
        std::size_t tgtlen = std::max(std::rand() % (std::min(tgtToken[pos].size(),sequenceLength - 1) + 1), 1ul);
        if(srclen > sequenceLength - 1) {
            --i;
            continue;
        }

        std::size_t* en = encoderInput + i * sequenceLength;
        std::size_t* de = decoderInput + i * sequenceLength;
        std::size_t* tr = tragetOutput + i * sequenceLength;

        for(int j = 0;j < srclen;j++) {
            en[j] = srcTokens[pos][j];
        }
        for(int j = srclen;j < sequenceLength;j++) {
            en[j] = 1.0f; // padding
        }

        de[0] = 2.0f; // sos
        for(int j = 1;j < tgtlen + 1;j++) {
            de[j] = tgtToken[pos][j - 1];
        }
        for(int j = tgtlen + 1;j < sequenceLength;j++) {
            de[j] = 1.0f; // padding
        }

        for(int j = 0;j < tgtlen;j++) {
            tr[j] = tgtToken[pos][j];
        }
        if(tgtToken[pos].size() <= tgtlen) {
            tr[tgtlen] = 3.0f; // eos
        }
        else {
            tr[tgtlen] = tgtToken[pos][tgtlen];
        }
        for(int j = tgtlen + 1;j < sequenceLength;j++) {
            tr[j] = 1.0f; // padding
        }

        srcSeq[i] = srclen;
        tgtSeq[i] = tgtlen;
    }
}