#include "Header.h"
#include "cnpy.h"
#include "Tensor.h"

Tensor::Tensor(const int row, const int col) : row(row), col(col) {
    data = new float[row * col];
}

Tensor::~Tensor() {
    delete[] data;
}

float& Tensor::operator[](const int idx) {
    return *(data + idx);
}

const float& Tensor::operator[](const int idx) const {
    return *(data + idx);
}

Tensor& Tensor::operator=(const float x) {
    for(int i = 0;i < row * col;i++) {
        data[i] = x;
    }
    return *this;
}

Tensor& Tensor::operator=(const TensorView other) {
    std::memcpy(data, other.data, sizeof(float) * row * col);
    return *this;
}

Tensor& Tensor::operator+=(const TensorView other) {
    for(int i = 0;i < row * col;i ++) {
        data[i] += other[i];
    }
    return *this;
}


TensorView TensorView::sliceRow(int r0,int r) {
    return TensorView(data + r0 * col, r, col);
}



TensorView::TensorView() : data(nullptr), row(0), col(0) {}

TensorView::TensorView(const TensorView& other)
    : data(other.data), row(other.row), col(other.col) {}

TensorView::TensorView(Tensor& t)
    : data(t.data), row(t.row), col(t.col) {}

TensorView::TensorView(float* data, int row, int col)
    : data(data), row(row), col(col) {}

float& TensorView::operator[](const int idx) {
    return *(data + idx);
}

const float& TensorView::operator[](const int idx) const {
    return *(data + idx);
}

TensorView& TensorView::operator=(const float x) {
    for(int i = 0;i < row * col;i++) {
        data[i] = x;
    }
    return *this;
}

TensorView& TensorView::operator=(const TensorView other) {
    std::memcpy(data, other.data, sizeof(float) * row * col);
    return *this;
}

TensorView& TensorView::operator+=(const TensorView other) {
    for(int i = 0;i < row * col;i ++) {
        data[i] += other[i];
    }
    return *this;
}


TensorView Tensor::sliceRow(int r0, int r) {
    return TensorView(data + r0 * col, r, col);
}

std::ofstream paramOutFile("../Param/out");
void Tensor::loadNp(cnpy::npz_t npFile, std::string name) {
    cnpy::NpyArray arr = npFile[name];
    std::memcpy(data, arr.data<float>(), sizeof(float) * row * col);
    for(int i = 0;i < row;i++) {
        for(int j = 0;j < col;j++) {
            paramOutFile << std::setprecision(9) << data[i * col + j] << " ";
        }
        paramOutFile << std::setprecision(9) <<  std::endl;
    }
    paramOutFile << std::setprecision(9) << std::endl;
}


void XavierUniformInit(TensorView A) {
    float limit = std::sqrt(6.0f / (A.row + A.col));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (int i = 0; i < A.row * A.col; i++) {
        A[i] = dist(gen);
    }
}

void UniformInit(TensorView A, const float limit) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (int i = 0; i < A.row * A.col; i++) {
        A[i] = dist(gen);
    }
}

void HeNormalInit(TensorView A) {
    std::random_device rd;
    std::mt19937 gen(rd());
    float stddev = std::sqrt(2.0f / A.row);
    std::normal_distribution<float> dist(0.0f, stddev);

    for (int i = 0; i < A.row * A.col; ++i) {
        A[i] = dist(gen);
    }
}

void Plus(TensorView A, TensorView B, TensorView C) {
    for(int i = 0;i < C.row * C.col;i++) {
        C[i] = A[i] + B[i];
    }
}

void Mul(TensorView A, TensorView B, TensorView C) {
    for(int i = 0;i < C.row * C.col;i++) {
        C[i] = A[i] * B[i];
    }
}

void Mul(TensorView A, const float B, TensorView C) {
    for(int i = 0;i < C.row * C.col;i++) {
        C[i] = A[i] * B;
    }
}

void Div(TensorView A, TensorView B, TensorView C) {
    for(int i = 0;i < C.row * C.col;i++) {
        C[i] = A[i] / B[i];
    }
}

void Div(TensorView A, const float B, TensorView C) {
    const float inv = 1.0f / B;
    Mul(A, inv, C);
}


void ApplyLookAheadMask(TensorView A, const int seq, const float x) {
    for(int i = 0;i < seq;i++) {
        for(int j = i + 1;j < sequenceLength;j++) {
            A[i * sequenceLength + j] = x;
        }
    }
    for(int i = seq;i < sequenceLength;i++){
        for(int j = 0;j < sequenceLength;j++) {
            A[i * sequenceLength + j] = x;
        }
    }
}
void ApplyPaddingMask(TensorView A, const int seq, const float x) {
    for(int i= 0 ;i < seq;i++) {
        for(int j = seq;j < sequenceLength;j++) {
            A[i * sequenceLength + j] = x;
        }
    }
    for(int i = seq;i < sequenceLength;i++) {
        for(int j = 0;j < sequenceLength;j++) {
            A[i * sequenceLength + j] = x;
        }
    }
}
void ApplyCrossPaddingMask(TensorView A, const int seq, const float x) {
    for(int i = 0;i < sequenceLength;i++) {
        for(int j = seq;j < sequenceLength;j++) {
            A[i * sequenceLength + j] = x;
        }
    }
}


void GetPositionalEncode(TensorView A) {
    for (int i = 0; i < sequenceLength; i++) {
        for (int j = 0; j < dModel; j += 2) {
            A[i*(dModel) + j] = std::sin(float(i) / std::pow(10000, float(j) / dModel));
        }
        for (int j = 1; j < dModel; j += 2) {
            A[i*(dModel) + j] = std::cos(float(i) / std::pow(10000, float(j - 1) / dModel));
        }
    }
}


void MatMulPlusAB(TensorView A, TensorView B, TensorView C) {
    const int d1 = C.row;
    const int d2 = A.col;
    const int d3 = C.col;
    for(int ii = 0;ii < d1;ii += BLOCK_SIZE) {
        for(int jj = 0;jj < d3;jj += BLOCK_SIZE) {
            for(int kk = 0;kk < d2;kk += BLOCK_SIZE) {

                for(int i = 0;i < BLOCK_SIZE && ii + i < d1;i++) {
                    for(int k = 0; k < BLOCK_SIZE && kk + k < d2;k++) {
                        float a = A[(ii + i) * d2 + (kk + k)];
                        for(int j = 0;j < BLOCK_SIZE && jj + j < d3;j++) {
                            C[(ii + i) * d3 + (jj + j)] += a * B[(kk + k) * d3 + (jj + j)];
                        }
                    }
                }

            }
        }
    }
}

void MatMulPlusATB(TensorView A, TensorView B, TensorView C) {
    const int d1 = C.row;
    const int d2 = A.row;
    const int d3 = C.col;
    for(int ii = 0;ii < d1;ii += BLOCK_SIZE) {
        for(int jj = 0;jj < d3;jj += BLOCK_SIZE) {
            for(int kk = 0;kk < d2;kk += BLOCK_SIZE) {

                for(int i = 0;i < BLOCK_SIZE && ii + i < d1;i++) {
                    for(int k = 0; k < BLOCK_SIZE && kk + k < d2;k++) {
                        float a = A[(kk + k) * d1 + (ii + i)];
                        for(int j = 0;j < BLOCK_SIZE && jj + j < d3;j++) {
                            C[(ii + i) * d3 + (jj + j)] += a * B[(kk + k) * d3 + (jj + j)];
                        }
                    }
                }

            }
        }
    }
}

void MatMulPlusABT(TensorView A, TensorView B, TensorView C) {
    Tensor BT(B.col, B.row);
    const int d2 = B.col;
    const int d3 = B.row;
    for(int ii = 0;ii < d2;ii += BLOCK_SIZE) {
        for(int jj = 0;jj < d3;jj += BLOCK_SIZE) {
            for(int i = 0;i < BLOCK_SIZE && ii + i < d2;i++){
                for(int j = 0;j < BLOCK_SIZE && jj + j < d3;j++) {
                    BT[(ii + i) * d3 + jj + j] = B[(jj + j) * d2 + ii + i];
                }
            }
        }
    }
    MatMulPlusAB(A, BT, C);
}