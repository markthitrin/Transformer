#include "Header.cuh"
#include "Transformer.cuh"
#include "Data.cuh"

void readDataFile(std::vector<std::vector<std::size_t>>& src, std::vector<std::vector<std::size_t>>& tgt) {
    std::ifstream file("opus_books_tokenized.txt");
    while(!file.eof()) {
        src.emplace_back();
        tgt.emplace_back();
        std::size_t N,M;
        file >> N;
        src.back().resize(N);
        for(int i = 0;i < N;i++) {
            file >> src.back()[i];
        }
        file >> M;
        tgt.back().resize(M);
        for(int i = 0;i < M;i++) {
            file >> tgt.back()[i];
        }
    }
    file.close();
}

void readTranslateFile(std::vector<std::string>& translator, const std::string& lang) {
    std::ifstream file("Translate_" + lang + ".txt");
    while(!file.eof()) {
        std::string buffer;
        file >> buffer;
        translator.push_back(std::move(buffer));
    }
    file.close();
}

void printSentence(std::vector<std::string>& translator, const std::size_t* token, const std::size_t* seq, std::size_t batchNum) {
    for(int i = 0;i < sequenceLength;i++) {
        std::cout << translator[token[batchNum * sequenceLength + i]] << " ";
        if(i == seq[batchNum] - 1) {
            std::cout << "| ";
        }
    }
    std::cout << std::endl;
}

void split(std::vector<std::vector<std::size_t>>& src, std::vector<std::vector<std::size_t>>& tgt,
    std::vector<std::vector<std::size_t>>& srcTrain, std::vector<std::vector<std::size_t>>& tgtTrain,
    std::vector<std::vector<std::size_t>>& srcTest, std::vector<std::vector<std::size_t>>& tgtTest,
    const float trainRatio = 0.7) {

    while(src.size()) {
        bool b = (float(std::rand() % 1000) / 1000) < trainRatio;
        if(b) {
            srcTrain.push_back(std::move(src.back()));
            tgtTrain.push_back(std::move(tgt.back()));
        } 
        else {
            srcTest.push_back(std::move(src.back()));
            tgtTest.push_back(std::move(tgt.back()));
        }
        src.pop_back();
        tgt.pop_back();
    }
}

const std::size_t* getOutputToken(Tensor& output) {
    static std::size_t* tokens = new std::size_t[output.row];
    static float* outputH = new float[output.row * output.col];
    output.toFloat(outputH);
    for(int i = 0;i < output.row;i++) {
        float maxValue = -FLT_MAX;
        float maxPos = 0;
        for(int j = 0;j < output.col;j++) {
            if(outputH[i * output.col + j] > maxValue) {
                maxValue = outputH[i * output.col + j];
                maxPos = j;
            }
        }
        tokens[i] = maxPos;
    }
    return tokens;
}

int main() {
    std::vector<std::vector<std::size_t>> src;
    std::vector<std::vector<std::size_t>> tgt;
    std::vector<std::vector<std::size_t>> srcTrain;
    std::vector<std::vector<std::size_t>> tgtTrain;
    std::vector<std::vector<std::size_t>> srcTest;
    std::vector<std::vector<std::size_t>> tgtTest;
    std::vector<std::string> translatorEn;
    std::vector<std::string> translatorIt;
    readDataFile(src, tgt);
    readTranslateFile(translatorEn, "en");
    readTranslateFile(translatorIt, "it");
    split(src, tgt, srcTrain, tgtTrain, srcTest, tgtTest);
    Data datasetTrain(std::move(srcTrain), std::move(tgtTrain));
    Data datasetTest(std::move(srcTest), std::move(tgtTest));

    Transformer model;



    {   // Training Section
        std::size_t* encoderInput = new std::size_t[batch * sequenceLength];
        std::size_t* srcSeq = new std::size_t[batch];
        std::size_t* decoderInput = new std::size_t[batch * sequenceLength];
        std::size_t* tgtSeq = new std::size_t[batch];
        std::size_t* targetOutput = new std::size_t[batch * sequenceLength];

        for(int i = 0;i < trainingIteration;i++) {
            datasetTrain.get(encoderInput, srcSeq, decoderInput, tgtSeq, targetOutput);
            float loss = model.Train(encoderInput, srcSeq, decoderInput, tgtSeq, targetOutput);
            std::cout << "Iteration [" << i << " / " << trainingIteration << "]   loss : " << loss << std::endl; 
        }
    }


    {   // Evaluation Section
        std::size_t* encoderInput = new std::size_t[batch * sequenceLength];
        std::size_t* srcSeq = new std::size_t[batch];
        std::size_t* decoderInput = new std::size_t[batch * sequenceLength];
        std::size_t* tgtSeq = new std::size_t[batch];
        std::size_t* targetOutput = new std::size_t[batch * sequenceLength];

        for(int i = 0;i < testingIteration;i++) {
            datasetTest.get(encoderInput, srcSeq, decoderInput, tgtSeq, targetOutput);
            model.Encode(encoderInput, srcSeq);
            model.Decode(decoderInput, tgtSeq);
            for(int j = 0;j < batch;j++) {
                std::cout << "English :::::::::::::::::\n";
                printSentence(translatorEn, encoderInput, srcSeq, j);
                std::cout << "Itelian (target) ::::::::\n";
                printSentence(translatorIt, targetOutput, tgtSeq, j);
                std::cout << "Itelian (predicted) ::::::::\n";
                printSentence(translatorIt, getOutputToken(model.output), tgtSeq, j);
                std::cout << std::endl << std::endl;;
            }
        }
    }
}