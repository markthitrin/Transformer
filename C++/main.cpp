#include "Header.h"
#include "Transformer.h"
#include "Data.h"
#include "Timer.h"

void readDataFile(std::vector<std::vector<int>>& src, std::vector<std::vector<int>>& tgt) {
    std::ifstream file("opus_books_tokenized.txt");
    while(!file.eof()) {
        src.emplace_back();
        tgt.emplace_back();
        int N,M;
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

void printSentence(std::vector<std::string>& translator, const int* token, const int* seq, int batchNum) {
    for(int i = 0;i < sequenceLength;i++) {
        std::cout << translator[token[batchNum * sequenceLength + i]] << " ";
        if(i == seq[batchNum] - 1) {
            break;
        }
    }
    std::cout << std::endl << std::endl;
}

void split(std::vector<std::vector<int>>& src, std::vector<std::vector<int>>& tgt,
    std::vector<std::vector<int>>& srcTrain, std::vector<std::vector<int>>& tgtTrain,
    std::vector<std::vector<int>>& srcTest, std::vector<std::vector<int>>& tgtTest,
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

const int* getOutputToken(Tensor& output) {
    static int* tokens = new int[output.row];
    for(int i = 0;i < output.row;i++) {
        float maxValue = -FLT_MAX;
        float maxPos = 0;
        for(int j = 0;j < output.col;j++) {
            if(output[i * output.col + j] > maxValue) {
                maxValue = output[i * output.col + j];
                maxPos = j;
            }
        }
        tokens[i] = maxPos;
    }
    return tokens;
}

int main() {
    std::vector<std::vector<int>> src;
    std::vector<std::vector<int>> tgt;
    std::vector<std::vector<int>> srcTrain;
    std::vector<std::vector<int>> tgtTrain;
    std::vector<std::vector<int>> srcTest;
    std::vector<std::vector<int>> tgtTest;
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
        int* encoderInput = new int[batch * sequenceLength];
        int* srcSeq = new int[batch];
        int* decoderInput = new int[batch * sequenceLength];
        int* tgtSeq = new int[batch];
        int* targetOutput = new int[batch * sequenceLength];
        Tensor output(batch * sequenceLength, tgtVocab);
        Tensor gradient(batch * sequenceLength, tgtVocab);

        for(int i = 0;i < trainingIteration;i++) {
            datasetTrain.get(encoderInput, decoderInput, targetOutput, srcSeq, tgtSeq);
            Timer::RestartRecord();
            model.forward(encoderInput, decoderInput, output, srcSeq, tgtSeq);
            float loss = CrossEntropy(output, targetOutput, tgtSeq, gradient);
            model.backward(gradient, encoderInput, decoderInput, srcSeq, tgtSeq);
            model.updateParameter();
            std::cout << "Iteration [" << i << " / " << trainingIteration << "]   loss : " << loss << std::endl; 
        }
    }

    std::cout << "Time Recorded ==========================\n\n";
    std::vector<double> record = Timer::GetTime();
    for(int q = 0;q < record.size();q++) {
        std::cout << record[q] << std::endl;
    }

    {   // Evaluation Section
        int* encoderInput = new int[batch * sequenceLength];
        int* srcSeq = new int[batch];
        int* decoderInput = new int[batch * sequenceLength];
        int* tgtSeq = new int[batch];
        int* targetOutput = new int[batch * sequenceLength];
        Tensor output(batch * sequenceLength, tgtVocab);

        for(int i = 0;i < testingIteration;i++) {
            datasetTest.get(encoderInput, decoderInput, targetOutput, srcSeq, tgtSeq);
            model.predict(encoderInput, decoderInput, output, srcSeq, tgtSeq);
            for(int j = 0;j < batch;j++) {
                std::cout << "English :::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n\n";
                printSentence(translatorEn, encoderInput, srcSeq, j);
                std::cout << "Italian (target) ::::::::::::::::::::::::::::::::::::::::::::::::\n\n";
                printSentence(translatorIt, targetOutput, tgtSeq, j);
                std::cout << "Italian (predicted) :::::::::::::::::::::::::::::::::::::::::::::\n\n";
                printSentence(translatorIt, getOutputToken(output), tgtSeq, j);
                std::cout << std::endl << std::endl << std::endl << std::endl;
            }
        }
    }
}