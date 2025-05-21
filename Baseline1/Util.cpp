#include "Header.h"

static float maxAbsChange = 0;

Matrix AdamOpt(Matrix& accM, Matrix& accV, const Matrix& gradient, const int t) {
	Matrix gradientPow2(gradient.row, gradient.column);
	for (int i = 0; i < gradient.data.size(); i++) {
		gradientPow2.data[i] = gradient.data[i] * gradient.data[i];
	}

	accM = accM * b1 + gradient * (1.0 - b1);
	accV = accV * b2 + gradientPow2 * (1.0 - b2);

	Matrix accMhat = accM / (1.0 - std::pow(b1,t));
	Matrix accVhat = accV / (1.0 - std::pow(b2,t));

	Matrix result(accMhat);
	for (int i = 0; i < result.data.size(); i++) {
		result.data[i] = result.data[i] / (std::sqrt(accVhat.data[i]) + eps);
	}

	float learningRate = dModel * std::min(std::pow(t, -0.5), t * std::pow(warmupStep, -1.5));
    /*for (int q = 0; q < result.row; q++) {
        for (int w = 0; w < result.column; w++) {
            if (std::abs(result[q][w]) > maxAbsChange) {
                maxAbsChange = std::abs(result[q][w]);
                std::cout << maxAbsChange << " :::: " << learningRate << std::endl;
            }
        }
    }*/
	return result * learningRate;
}

float CrossEntropy(const Tensor& output, const Tensor& target) {
    float loss = 0;
    for (int i = 0; i < output.batch; i++) {
        int targetToken = target[i][0][0];
        loss += std::log(output[i][targetToken][0]);
    }
    loss *= -1.0f / (output.batch);
    return loss;
}

Tensor CrossEntropyGradient(const Tensor& output, const Tensor& target) {
    Tensor result(output.batch, output.data[0].row, 1);
    for (int i = 0; i < output.batch; i++) {
        int targetToken = target[i][0][0];
        if (output[i][targetToken][0] < 1e-8) {
            result[i][targetToken][0] = -1.0f / 1e-8 / output.batch;
        }
        else {
            result[i][targetToken][0] = -1.0f / output[i][targetToken][0] / output.batch;
        }
    }
    return result;
}

void XavierUniformInit(Matrix& W) {
    int n_in = W.row;
    int n_out = W.column;
    float limit = std::sqrt(6.0f / (n_in + n_out));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (int i = 0; i < n_in; ++i) {
        for (int j = 0; j < n_out; ++j) {
            W[i][j] = dist(gen);
        }
    }
}

void UniformInit(Matrix& W, const float limit) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (int i = 0; i < W.row; ++i) {
        for (int j = 0; j < W.column; ++j) {
            W[i][j] = dist(gen);
        }
    }
}

void HeNormalInit(Matrix& W) {
    int n_in = W.row;
    std::random_device rd;
    std::mt19937 gen(rd());
    float stddev = std::sqrt(2.0f / n_in);
    std::normal_distribution<float> dist(0.0f, stddev);

    for (int i = 0; i < W.row; ++i) {
        for (int j = 0; j < W.column; ++j) {
            W[i][j] = dist(gen);
        }
    }
}

int RandomInt(int low, int high) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(low, high);
    return dist(gen);
}