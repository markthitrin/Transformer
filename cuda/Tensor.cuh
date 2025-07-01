#ifndef TENSOR
#define TENSOR

#include "Header.cuh"
#include "cnpy.h"

class Tensor{
public:
    Tensor() noexcept;
    Tensor(const Tensor& other) noexcept;
    Tensor(Tensor&& other) noexcept;
    Tensor(const std::size_t row,const std::size_t col) noexcept;
    ~Tensor() noexcept;

    void free() noexcept;

    void toFloat(float* _data) noexcept;

    void XavierUniformFill() noexcept;

    void UniformFill(const float limit) noexcept;

    void HeNormalFill() noexcept;
    
    float* data;
    std::size_t pitch;
    std::size_t row;
    std::size_t col;
};

#endif