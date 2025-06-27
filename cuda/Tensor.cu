#include "Header.cuh"
#include "Tensor.cuh"
#include "TensorFunction.cuh"

Tensor::Tensor() noexcept {;}

Tensor::Tensor(const Tensor& other) noexcept : pitch(other.pitch), row(other.row), col(other.col) {
    cudaError_t err = cudaMallocPitch(&data, &pitch, col * sizeof(float), row);
    PRINT_CUDA_ERR(err);
    cudaMemcpy2D(data, pitch, other.data, other.pitch, sizeof(float) * col, row, cudaMemcpyDeviceToDevice);
}

Tensor::Tensor(Tensor&& other) noexcept : data(other.data), pitch(other.pitch), row(other.row), col(other.col) {
    other.data = nullptr;
}
    
Tensor::Tensor(const std::size_t row,const std::size_t col) noexcept : row(row), col(col)  {
    cudaError_t err = cudaMallocPitch(&data, &pitch, col * sizeof(float), row);
    PRINT_CUDA_ERR(err);
}

Tensor::~Tensor() {
    cudaFree(data);
}

void Tensor::free() noexcept {
    cudaFree(data);
}

void Tensor::toFloat(float* _data) noexcept {
    cudaError_t err = cudaMemcpy2D(_data, sizeof(float) * col, data, pitch, sizeof(float) * col, row, cudaMemcpyDeviceToHost);
    PRINT_CUDA_ERR(err);
}

void Tensor::loadNp(const cnpy::npz_t& npFile, const std::string& name) noexcept {
    cnpy::NpyArray arr = npFile.at(name);
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