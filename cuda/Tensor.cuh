#ifndef TENSOR
#define TENSOR

#include "Header.cuh"
#include "cnpy.h"

class Tensor{
public:
    Tensor() noexcept;
    Tensor(Tensor& other) noexcept;
    Tensor(const std::size_t row,const std::size_t col) noexcept;

    void free();

    void toFloat(float* _data);

    void loadNp(cnpy::npz_t npFile, std::string name);

    void XavierUniformFill();

    void UniformFill(const float limit);

    void HeNormalFill();
    
    float* data;
    std::size_t pitch;
    std::size_t row;
    std::size_t col;
};

#endif