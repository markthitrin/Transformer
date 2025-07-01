#ifndef DATA
#define DATA

#include "Header.cuh"
#include "Config.h"

class Data {
public:
    Data(std::vector<std::vector<std::size_t>>&& srcTokens, std::vector<std::vector<std::size_t>>&& tgtToken) noexcept;
    
    void get(std::size_t* encoderInput, std::size_t* srcSeq,
        std::size_t* decoderInput, std::size_t* tgtSeq,
        std::size_t* tragetOutput);

    std::vector<std::vector<std::size_t>> srcTokens;
    std::vector<std::vector<std::size_t>> tgtToken;
};

#endif