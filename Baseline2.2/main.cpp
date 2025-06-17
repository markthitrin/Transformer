#include <iostream>
#include "Header.h"
#include "Transformer.h"
#include "Data.h"

void readFile(std::ifstream& file, std::vector<std::vector<float>>& s,std::vector<std::vector<float>>& t) {
    while(!file.eof()) {
        int N,M;
        file >> N;
        s.push_back(std::vector<float>(N));
        for(int q = 0;q < N;q++) {
            file >> s.back()[q];
        }
        file >> M;
        t.push_back(std::vector<float>(M));
        for(int q = 0;q < M;q++) {
            file >> t.back()[q];
        }
    }
}

void readTranslate(std::ifstream& file,std::vector<std::string>& text) {
    while(!file.eof()) {
        std::string s;
        file >> s;
        text.push_back(s);
    }
}

int main() {
    std::vector<std::vector<float>> source;
    std::vector<std::vector<float>> target;
    std::vector<std::string> translator;
    std::ifstream dataFile("opus_books_tokenized.txt");
    std::ifstream translation("Translate.txt");
    readFile(dataFile, source, target);
    readTranslate(translation, translator);
    const int traningSize = source.size() * trainingRatio;
    
    std::vector<std::vector<float>> trainingSource;
    std::vector<std::vector<float>> trainingTraget;
    std::vector<std::vector<float>> testingSource;
    std::vector<std::vector<float>> testingTraget;
    for(int i = 0;i < traningSize;i++) {
        trainingSource.push_back(std::move(source[i]));
        trainingTraget.push_back(std::move(target[i]));
    }
    for(int i = traningSize;i < source.size();i++) {
        testingSource.push_back(std::move(source[i]));
        testingTraget.push_back(std::move(target[i]));
    }

    Data<batch, sequenceLength> datasetTraining(std::move(trainingSource), std::move(trainingTraget));
    Data<batch, sequenceLength> datsetTesting(std::move(testingSource), std::move(testingTraget));
    Transformer model;
    
    // Training
    Tensor<1 ,batch * sequenceLength> encoderInput, decoderInput, decoderTarget;
    Tensor<batch * sequenceLength, tgtVocab> output, gradient;
    encoderInput.init();
    decoderInput.init();
    decoderTarget.init();
    output.init();
    gradient.init();
    model._inputEncoder = encoderInput;
    model._inputDecoder = decoderInput;
    model._output = output;
    model._inGradient = gradient; 
    int npdSrc[batch], npdTgt[batch];
    for(int q = 0;q < 10000;q++) {
        datasetTraining.get(encoderInput, decoderInput, decoderTarget, npdSrc, npdTgt);
        model.forward(npdSrc, npdTgt);
        float loss = CrossEntropy(output, decoderTarget, gradient, npdTgt);
        model.backward(npdSrc, npdTgt);
        model.updateParameter();
        std::cout << "Iteration : " << q + 1 << " / 1000 --- Loss : " << loss << std::endl;
    }

    int outputToken[batch * sequenceLength];
    for(int q = 0;q < 10;q++) {
        datsetTesting.get(encoderInput, decoderInput, decoderTarget, npdSrc, npdTgt);
        model.predict(npdSrc, npdTgt);
        GetAnswer(output, outputToken);
        for(int i = 0;i < batch;i++) {
            float* rowTar = decoderTarget.data + i * sequenceLength;
            int* rowOut = outputToken + i * sequenceLength;
            std::cout << "===========================================================\n";
            std::cout << "target : ";
            for(int j = 0; j < sequenceLength;j++) {
                std::cout << translator[(int)rowTar[j]] << " ";
            }
            std::cout << std::endl;
            std::cout << "output : ";
            for(int j = 0;j < sequenceLength;j++) {
                std::cout << translator[rowOut[j]] << " ";
            }
            std::cout << std::endl;
            std::cout << std::endl;
        }
    }
}