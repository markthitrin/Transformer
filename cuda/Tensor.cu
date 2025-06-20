#include "Header.cuh"
#include "Tensor.cuh"

Tensor::Tensor() noexcept {;}

Tensor::Tensor(Tensor& other) noexcept : data(other.data), pitch(other.pitch), row(other.row), col(other.col) {;}
    
Tensor::Tensor(const std::size_t row,const std::size_t col) noexcept : row(row), col(col)  {
    cudaMallocPitch(&data, &pitch, col * sizeof(float), row);
}

void Tensor::free() noexcept {
    cudaFree(data);
}

void Tensor::toFloat(float* _data) noexcept {
    cudaMemcpy2D(_data, sizeof(float) * col, data, pitch, sizeof(float) * col, row, cudaMemcpyDeviceToHost);
}

void Tensor::loadNp(cnpy::npz_t npFile, std::string name) noexcept {
    cnpy::NpyArray arr = npFile[name];
    cudaMemcpy2D(data, pitch, arr.data<float>(), sizeof(float) * col, sizeof(float) * col, row, cudaMemcpyHostToDevice);
}

void Tensor::XavierUniformFill() noexcept {
    float* _data = (float*)malloc(sizeof(float) * row * col);
    float limit = std::sqrt(6.0f / (row + col));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (int i = 0; i < row * col; i++) {
        _data[i] = dist(gen);
    }
    cudaMemcpy2D(data, pitch, _data, sizeof(float) * col, sizeof(float) * col, row, cudaMemcpyHostToDevice);
    std::free(_data);
}

void Tensor::UniformFill(const float limit) noexcept {
    float* _data = (float*)malloc(sizeof(float) * row * col);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (int i = 0; i < row * col; i++) {
        _data[i] = dist(gen);
    }
    cudaMemcpy2D(data, pitch, _data, sizeof(float) * col, sizeof(float) * col, row, cudaMemcpyHostToDevice);
    std::free(_data);
}

void Tensor::HeNormalFill() noexcept {
    float* _data = (float*)malloc(sizeof(float) * row * col);
    std::random_device rd;
    std::mt19937 gen(rd());
    float stddev = std::sqrt(2.0f / row);
    std::normal_distribution<float> dist(0.0f, stddev);

    for (int i = 0; i < row * col; ++i) {
        _data[i] = dist(gen);
    }
    cudaMemcpy2D(data, pitch, _data, sizeof(float) * col, sizeof(float) * col, row, cudaMemcpyHostToDevice);
    std::free(_data);
}