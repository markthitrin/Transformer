#ifndef TENSOR
#define TENSOR

#include "Header.h"
#include "cnpy.h"

class Tensor;

class TensorView {
public:
    TensorView();
    TensorView(const TensorView& other);
    TensorView(Tensor& t);
    TensorView(float* data, const int row, const int col);

    float& operator[](const int idx);
    const float& operator[](const int idx) const;
    TensorView& operator=(const float x);
    TensorView& operator=(const TensorView other);
    TensorView& operator+=(const TensorView other);

    TensorView sliceRow(int r0,int r);

    float* data;
    int row;
    int col;
};

class Tensor {
public:
    Tensor() : data(nullptr) {;}
    Tensor(const int row, const int col);
    ~Tensor();

    float& operator[](const int idx);
    const float& operator[](const int idx) const;
    Tensor& operator=(const float x);
    Tensor& operator=(const TensorView other);
    Tensor& operator+=(const TensorView other);

    TensorView sliceRow(int r0,int r);
    void loadNp(cnpy::npz_t npFile, std::string name);

    float* data;
    int row;
    int col;
};


void XavierUniformInit(TensorView A);

void UniformInit(TensorView A, const float limit);

void HeNormalInit(TensorView A);


void Copy(TensorView A, TensorView B);

void Plus(TensorView A, TensorView B, TensorView C);

void Mul(TensorView A, TensorView B, TensorView C);
void Mul(TensorView A, const float B, TensorView C);

void Div(TensorView A, TensorView B, TensorView C);
void Div(TensorView A, const float B, TensorView C);


void ApplyLookAheadMask(TensorView A, const int seq, const float x);
void ApplyPaddingMask(TensorView A, const int seq, const float x);
void ApplyCrossPaddingMask(TensorView A, const int seqv, const float x);

void GetPositionalEncode(TensorView A);

void MatMulPlusAB(TensorView A, TensorView B, TensorView C);

void MatMulPlusATB(TensorView A, TensorView B, TensorView C);

void MatMulPlusABT(TensorView A, TensorView B, TensorView C);

#endif