#include "Header.h"

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
	int numToken = 0;
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
	for (int q = 0; q < dataToken.size(); q++) {
		numToken = std::max(numToken, dataToken[q] + 1);
	}
	Data trainData(trainDataToken);
	Data testData(testDataToken);
	dataToken.clear();
	trainDataToken.clear();
	testDataToken.clear();

	Decoder model(512, 6, numToken);

	for (int i = 0; i < 1000; i++) {
		std::pair<Tensor, Tensor> data = trainData.getData(batch, 17);
		Tensor output = model(sequenceLength , data.first);
		float loss = CrossEntropy(output, data.second);
		Tensor gradient = CrossEntropyGradient(output, data.second);
		model.backpropagate(sequenceLength, gradient);
		model.updateParameter();
		std::cout << "teration : " << i + 1 << " / 10000" << "  loss :: : " << loss << std::endl;
	}
	
	for (int i = 0; i < 100; i++) {
		std::pair<Tensor, Tensor> data = testData.getData(batch, 17);
		Tensor target = data.second;
		Tensor output = model.predict(sequenceLength, data.first);
		int batch = output.batch / sequenceLength;
		for (int j = 0; j < batch; j++) {
			std::cout << "predict :::::::::::::::::::::::::::::::::::::::::::::::::;" << std::endl;
			int s = j * sequenceLength;
			std::cout << "output :: ";
			for (int k = sequenceLength - 1; k < sequenceLength; k++) {
				float maxValue = 0;
				float maxInd = 0;
				for (int w = 0; w < output.data[0].row; w++) {
					if (maxValue < output[s + k][w][0]) {
						maxValue = output[s + k][w][0];
						maxInd = w;
					}
				}
				std::cout << "[" << dataTranslate[maxInd] << "]" << " {" << int(dataTranslate[maxInd]) << "}";
			}
			std::cout << std::endl;
			std::cout << "target :: ";
			for (int k = sequenceLength - 1; k < sequenceLength; k++) {
				std::cout << "[" << dataTranslate[(int(target[s + k][0][0]))] << "]" << " {" << int(dataTranslate[(int(target[s + k][0][0]))]) << "}";
			}
			std::cout << std::endl;
			std::cout << std::endl;
		}
	}

	return 0;
}