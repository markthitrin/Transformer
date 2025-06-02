#include "Header.h"
#include "Decoder.h"
#include "Data.h"

void tokenize() {
	std::ifstream file("DatasetRaw.txt");
	std::ofstream outFile("DatasetToken.txt");
	std::ofstream outFile2("DatasetTranslate.txt");
	std::string data;
	while (!file.eof()) {
		std::string inLine;
		std::getline(file, inLine);
		data += inLine + '\n';
	}

	std::map<char, int> s;
	int count = 1;
	for (int q = 0; q < data.size(); q++) {
		if (!s.count(data[q])) {
			s[data[q]] = count++;
		}
		outFile << s[data[q]] << "\n";
	}
	std::vector<int> tokenChar(count);
	for (auto it = s.begin(); it != s.end(); it++) {
		tokenChar[it->second] = it->first;
	}
	for (int q = 0; q < tokenChar.size(); q++) {
		outFile2 << tokenChar[q] << "\n";
	}
	file.close();
	outFile.close();
	outFile2.close();
}

std::vector<int> getToken() {
	std::vector<int> result;
	std::ifstream file("DatasetToken.txt");
	while (!file.eof()) {
		int input;
		file >> input;
		result.push_back(input);
	}
	return result;
}

std::vector<char> getTranslate() {
	std::vector<char> result;
	std::ifstream file("DatasetTranslate.txt");
	while (!file.eof()) {
		int input;
		file >> input;
		result.push_back(char(input));
	}
	return result;
}

int main() {
	constexpr int numToken = 128;
	std::vector<int> dataToken = getToken();
	std::vector<char> dataTranslate = getTranslate();
	std::vector<int> trainDataToken;
	std::vector<int> testDataToken;
	int i = 0;
	while (i < 0.7 * dataToken.size()) {
		trainDataToken.push_back(dataToken[i]);
		++i;
	}
	while (i < dataToken.size()) {
		testDataToken.push_back(dataToken[i]);
		++i;
	}
	
	Data trainData(trainDataToken);
	Data testData(testDataToken);
	dataToken.clear();
	trainDataToken.clear();
	testDataToken.clear();

	Decoder<batch,numToken, sequenceLength,dModel,numToken> model;
	Tensor input = Create0<1, batch* sequenceLength>();
	Tensor target = Create0<1, batch* sequenceLength>();
	Tensor output = Create0<batch * sequenceLength, numToken>();
	Tensor gradient = Create0<batch * sequenceLength, numToken>();
	model._input = input;
	model._output = output;
	model._inGradient = gradient;

	for (int i = 0; i < 1000; i++) {
		trainData.getData<batch, sequenceLength>(input, target);
		model.forward();
		float loss = CrossEntropy<batch * sequenceLength, numToken>(output, target, gradient);
		model.backpropagate();
		model.updateParameter();

		std::cout << "teration : " << i + 1 << " / 10000" << "  loss :: : " << loss << std::endl;
	}

	for (int i = 0; i < 5; i++) {
		trainData.getData<batch, sequenceLength>(input, target);
		model.predict();
		for (int j = 0; j < batch; j++) {
			std::cout << "predict ::::::::::::::::::::::::::::::::::::::::::::::::::" << std::endl;
			for (int k = 0; k < sequenceLength; k++) {
				const int s = j * sequenceLength * numToken + k * numToken;
				float maxValue = 0;
				float maxInd = 0;
				for (int w = 0; w < numToken; w++) {
					if (maxValue < output[s + w]) {
						maxValue = output[s + w];
						maxInd = w;
					}
				}
				std::cout << "--output--[" << dataTranslate[maxInd] << "] {" << int(dataTranslate[maxInd]) << "}\n";
				std::cout << "--target--[" << dataTranslate[int(target[j * sequenceLength + k])] << "] {" << int(dataTranslate[int(target[j * sequenceLength + k])]) << "}\n";
			}
		}
		Reset<batch* sequenceLength, numToken>(output);
		Reset<1, batch* sequenceLength>(input);
		Reset<1, batch* sequenceLength>(target);
		Reset<batch* sequenceLength, numToken>(gradient);
	}

	return 0;
}