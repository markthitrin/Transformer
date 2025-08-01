#ifndef TENSOR
#define TENSOR

#include "Config.h"
#include "Header.h"
#include "cnpy.h"

class Tensor;

class TensorView {
public:
    inline TensorView();
    inline TensorView(const TensorView& other);
    inline TensorView(Tensor& t);
    inline TensorView(float* data, int row, int col);

    inline float& operator[](const int idx);
    inline const float& operator[](const int idx) const;
    
    inline TensorView& operator=(const float x);
    inline TensorView& operator=(const TensorView other);
    inline TensorView& operator+=(const TensorView other);

    inline TensorView sliceRow(int r0,int r);

    float* data;
    int row;
    int col;
};

class Tensor {
public:
    inline Tensor();
    inline Tensor(const int row, const int col);
    inline ~Tensor();

    inline float& operator[](const int idx);
    inline const float& operator[](const int idx) const;
    inline Tensor& operator=(const float x);
    inline Tensor& operator=(const TensorView other);
    inline Tensor& operator+=(const TensorView other);
    inline TensorView sliceRow(int r0,int r);

    float* data;
    int row;
    int col;
};


inline TensorView::TensorView() : data(nullptr), row(0), col(0) {}

inline TensorView::TensorView(const TensorView& other)
    : data(other.data), row(other.row), col(other.col) {}

inline TensorView::TensorView(Tensor& t)
    : data(t.data), row(t.row), col(t.col) {}

inline TensorView::TensorView(float* data, int row, int col)
    : data(data), row(row), col(col) {}

inline float& TensorView::operator[](const int idx) {
    return *(data + idx);
}

inline const float& TensorView::operator[](const int idx) const {
    return *(data + idx);
}

inline TensorView& TensorView::operator=(const float x) {
    for(int i = 0;i < row * col;i++) {
        data[i] = x;
    }
    return *this;
}

inline TensorView& TensorView::operator=(const TensorView other) {
    std::memcpy(data, other.data, sizeof(float) * row * col);
    return *this;
}

inline TensorView& TensorView::operator+=(const TensorView other) {
    for(int i = 0;i < row * col;i ++) {
        data[i] += other[i];
    }
    return *this;
}

inline TensorView TensorView::sliceRow(int r0,int r) {
    return TensorView(data + r0 * col, r, col);
}







inline Tensor::Tensor() : data(nullptr) {;}
inline Tensor::Tensor(const int row, const int col) : row(row), col(col) {
    data = new float[row * col];
}

inline Tensor::~Tensor() {
    delete[] data;
}

inline float& Tensor::operator[](const int idx) {
    return *(data + idx);
}

inline const float& Tensor::operator[](const int idx) const {
    return *(data + idx);
}

inline Tensor& Tensor::operator=(const float x) {
    for(int i = 0;i < row * col;i++) {
        data[i] = x;
    }
    return *this;
}

inline Tensor& Tensor::operator=(const TensorView other) {
    std::memcpy(data, other.data, sizeof(float) * row * col);
    return *this;
}

inline Tensor& Tensor::operator+=(const TensorView other) {
    for(int i = 0;i < row * col;i ++) {
        data[i] += other[i];
    }
    return *this;
}

inline TensorView Tensor::sliceRow(int r0,int r) {
    return TensorView(data + r0 * col, r, col);
}





inline void CopyPar(TensorView A, TensorView B) {
    const int numT = std::min(numPar, 24);
    #pragma omp parallel num_threads(numT)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        int chunk_size = (A.row + nthreads - 1) / nthreads;
        int start = tid * chunk_size;
        int end = std::min(start + chunk_size, A.row);

        B.sliceRow(start, end - start) = A.sliceRow(start, end - start);
    }
}

inline void SetPar(TensorView A, const float x) {
    const int numT = std::min(numPar, 24);
    #pragma omp parallel num_threads(numT)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        int chunk_size = (A.row + nthreads - 1) / nthreads;
        int start = tid * chunk_size;
        int end = std::min(start + chunk_size, A.row);

        A.sliceRow(start, end - start) = x;
    }
}


inline void XavierUniformInit(TensorView A) {
    float limit = std::sqrt(6.0f / (A.row + A.col));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (int i = 0; i < A.row * A.col; i++) {
        A[i] = dist(gen);
    }
}

inline void UniformInit(TensorView A, const float limit) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (int i = 0; i < A.row * A.col; i++) {
        A[i] = dist(gen);
    }
}

inline void HeNormalInit(TensorView A) {
    std::random_device rd;
    std::mt19937 gen(rd());
    float stddev = std::sqrt(2.0f / A.row);
    std::normal_distribution<float> dist(0.0f, stddev);

    for (int i = 0; i < A.row * A.col; ++i) {
        A[i] = dist(gen);
    }
}

inline void Plus(TensorView A, TensorView B, TensorView C) {
    for(int i = 0;i < C.row * C.col;i++) {
        C[i] = A[i] + B[i];
    }
}

inline void PlusPar(TensorView A, TensorView B, TensorView C) {
    const int numT = std::min(numPar, 24);
    #pragma omp parallel for num_threads(numT) schedule(static)
    for(int i = 0;i < C.row * C.col;i++) {
        C[i] = A[i] + B[i];
    }
}

inline void Mul(TensorView A, TensorView B, TensorView C) {
    for(int i = 0;i < C.row * C.col;i++) {
        C[i] = A[i] * B[i];
    }
}

inline void MulPar(TensorView A, TensorView B, TensorView C) {
    const int numT = std::min(numPar, 24);
    #pragma omp parallel for num_threads(numT) schedule(static)
    for(int i = 0;i < C.row * C.col;i++) {
        C[i] = A[i] * B[i];
    }
}

inline void Mul(TensorView A, const float B, TensorView C) {
    for(int i = 0;i < C.row * C.col;i++) {
        C[i] = A[i] * B;
    }
}

inline void MulPar(TensorView A, const float B, TensorView C) {
    const int numT = std::min(numPar, 24);
    #pragma omp parallel for num_threads(numT) schedule(static)
    for(int i = 0;i < C.row * C.col;i++) {
        C[i] = A[i] * B;
    }
}

inline void Div(TensorView A, TensorView B, TensorView C) {
    for(int i = 0;i < C.row * C.col;i++) {
        C[i] = A[i] / B[i];
    }
}

inline void DivPar(TensorView A, TensorView B, TensorView C) {
    const int numT = std::min(numPar, 24);
    #pragma omp parallel for num_threads(numT) schedule(static)
    for(int i = 0;i < C.row * C.col;i++) {
        C[i] = A[i] / B[i];
    }
}


inline void Div(TensorView A, const float B, TensorView C) {
    const float inv = 1.0f / B;
    Mul(A, inv, C);
}

inline void DivPar(TensorView A, const float B, TensorView C) {
    const float inv = 1.0f / B;
    MulPar(A, inv, C);
}


inline void ApplyLookAheadMask(TensorView A, const int seq, const float x) {
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

inline void ApplyPaddingMask(TensorView A, const int seq, const float x) {
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

inline void ApplyCrossPaddingMask(TensorView A, const int seq, const float x) {
    for(int i = 0;i < sequenceLength;i++) {
        for(int j = seq;j < sequenceLength;j++) {
            A[i * sequenceLength + j] = x;
        }
    }
}

inline void GetPositionalEncode(TensorView A) {
    for (int i = 0; i < sequenceLength; i++) {
        for (int j = 0; j < dModel; j += 2) {
            A[i*(dModel) + j] = std::sin(float(i) / std::pow(10000, float(j) / dModel));
        }
        for (int j = 1; j < dModel; j += 2) {
            A[i*(dModel) + j] = std::cos(float(i) / std::pow(10000, float(j - 1) / dModel));
        }
    }
}

inline void MatMulPlusAB(TensorView A, TensorView B, TensorView C) {
    const int d1 = C.row;
    const int d2 = A.col;
    const int d3 = C.col;
    for(int ii = 0;ii < d1;ii += BLOCK_SIZE) {
        for(int jj = 0;jj < d3;jj += BLOCK_SIZE) {
            for(int kk = 0;kk < d2;kk += BLOCK_SIZE) {

                for(int i = 0;(i < BLOCK_SIZE) & (ii + i < d1);i++) {
                    for(int k = 0; (k < BLOCK_SIZE ) & (kk + k < d2);k++) {
                        for(int j = 0;(j < BLOCK_SIZE) & (jj + j < d3);j++) {
                            C[(ii + i) * d3 + (jj + j)] += A[(ii + i) * d2 + (kk + k)] * B[(kk + k) * d3 + (jj + j)];
                        }
                    }
                }

            }
        }
    }
}

inline void MatMulPlusABPar(TensorView A, TensorView B, TensorView C) {
    const int d1 = C.row;
    const int d2 = A.col;
    const int d3 = C.col;
    #pragma omp parallel for collapse(2) schedule(static)
    for(int ii = 0;ii < d1;ii += BLOCK_SIZE) {
        for(int jj = 0;jj < d3;jj += BLOCK_SIZE) {
            for(int kk = 0;kk < d2;kk += BLOCK_SIZE) {

                for(int i = 0;(i < BLOCK_SIZE) & (ii + i < d1);i++) {
                    for(int k = 0; (k < BLOCK_SIZE ) & (kk + k < d2);k++) {
                        for(int j = 0;(j < BLOCK_SIZE) & (jj + j < d3);j++) {
                            C[(ii + i) * d3 + (jj + j)] += A[(ii + i) * d2 + (kk + k)] * B[(kk + k) * d3 + (jj + j)];
                        }
                    }
                }

            }
        }
    }
}

inline void MatMulPlusATB(TensorView A, TensorView B, TensorView C) {
    const int d1 = C.row;
    const int d2 = A.row;
    const int d3 = C.col;
    for(int ii = 0;ii < d1;ii += BLOCK_SIZE) {
        for(int jj = 0;jj < d3;jj += BLOCK_SIZE) {
            for(int kk = 0;kk < d2;kk += BLOCK_SIZE) {

                for(int i = 0;(i < BLOCK_SIZE) & (ii + i < d1);i++) {
                    for(int k = 0; (k < BLOCK_SIZE) & (kk + k < d2);k++) {
                        for(int j = 0;(j < BLOCK_SIZE) & (jj + j < d3);j++) {
                            C[(ii + i) * d3 + (jj + j)] += A[(kk + k) * d1 + (ii + i)] * B[(kk + k) * d3 + (jj + j)];
                        }
                    }
                }

            }
        }
    }
}

inline void MatMulPlusATBPar(TensorView A, TensorView B, TensorView C) {
    const int d1 = C.row;
    const int d2 = A.row;
    const int d3 = C.col;
    #pragma omp parallel for collapse(2) schedule(static)
    for(int ii = 0;ii < d1;ii += BLOCK_SIZE) {
        for(int jj = 0;jj < d3;jj += BLOCK_SIZE) {
            for(int kk = 0;kk < d2;kk += BLOCK_SIZE) {
                
                for(int i = 0;(i < BLOCK_SIZE) & (ii + i < d1);i++) {
                    for(int k = 0; (k < BLOCK_SIZE) & (kk + k < d2);k++) {
                        for(int j = 0;(j < BLOCK_SIZE) & (jj + j < d3);j++) {
                            C[(ii + i) * d3 + (jj + j)] += A[(kk + k) * d1 + (ii + i)] * B[(kk + k) * d3 + (jj + j)];
                        }
                    }
                }

            }
        }
    }
}

inline void MatMulPlusABT(TensorView A, TensorView B, TensorView C) {
    Tensor BT(B.col, B.row);
    const int d2 = B.col;
    const int d3 = B.row;
    for(int ii = 0;ii < d2;ii += BLOCK_SIZE) {
        for(int jj = 0;jj < d3;jj += BLOCK_SIZE) {
            for(int i = 0;(i < BLOCK_SIZE) & (ii + i < d2);i++){
                for(int j = 0;(j < BLOCK_SIZE) & (jj + j < d3);j++) {
                    BT[(ii + i) * d3 + jj + j] = B[(jj + j) * d2 + ii + i];
                }
            }
        }
    }
    MatMulPlusAB(A, BT, C);
}

inline void MatMulPlusABTPar(TensorView A, TensorView B, TensorView C) {
    Tensor BT(B.col, B.row);
    const int d2 = B.col;
    const int d3 = B.row;
    #pragma omp parallel for collapse(2) schedule(static)
    for(int ii = 0;ii < d2;ii += BLOCK_SIZE) {
        for(int jj = 0;jj < d3;jj += BLOCK_SIZE) {
            for(int i = 0;(i < BLOCK_SIZE) & (ii + i < d2);i++){
                for(int j = 0;(j < BLOCK_SIZE) & (jj + j < d3);j++) {
                    BT[(ii + i) * d3 + jj + j] = B[(jj + j) * d2 + ii + i];
                }
            }
        }
    }
    MatMulPlusABPar(A, BT, C);
}


#endif