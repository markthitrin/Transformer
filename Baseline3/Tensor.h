#ifndef TENSOR
#define TENSOR

#include "Header.h"
#include "cnpy.h"
#include <mkl.h>
#include <stdexcept>

constexpr int GetSizeBytes(int row,int col) {
    return row * col * sizeof(float);
}

template<int row, int col>
float* Create() {
    constexpr int realSize = GetSizeBytes(row, col);
    void* data = std::aligned_alloc(32, realSize);
    std::memset(data, 0, realSize);
    return (float*)data;
}

template<int row,int col>
class Tensor {
public:
    Tensor() : data(nullptr) {;}
    Tensor(Tensor<row, col>& other) : data(other.data) {;}
    Tensor(float* ptr) : data(ptr) {;}

    void free() {
        std::free(data);
    }

    void set(float* ptr) {
        if(data) std::free(data);
        data = ptr;
    }

    void init() {
        if(data) std::free(data);
        data = Create<row, col>();
    }

    template<int _row>
    Tensor<_row, col> sliceRow(int r0) {
        return Tensor<_row, col>(data + r0*(col));
    }

    void XavierUniformInit() {
        init();
        float limit = std::sqrt(6.0f / (row + col));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-limit, limit);

        for (int i = 0; i < row * col; i++) {
            data[i] = dist(gen);
        }
    }

    void UniformInit(const float limit) {
        init();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-limit, limit);

        for (int i = 0; i < row * col; i++) {
            data[i] = dist(gen);
        }
    }

    void HeNormalInit() {
        init();
        std::random_device rd;
        std::mt19937 gen(rd());
        float stddev = std::sqrt(2.0f / row);
        std::normal_distribution<float> dist(0.0f, stddev);

        for (int i = 0; i < row * col; ++i) {
            data[i] = dist(gen);
        }
    }

    float* data;
};

template<int row,int col>
void Copy(Tensor<row, col> _A, Tensor<row, col> _B) {
    IMPORT_CONST(A);
    IMPORT(B);

    constexpr int realSize = GetSizeBytes(row, col);
    std::memcpy((void*)B, (void*)A, realSize);
}

template<int row, int col>
void Plus(Tensor<row, col> _A, Tensor<row, col> _B, Tensor<row, col> _C) {
    IMPORT_CONST(A);
    IMPORT_CONST(B);
    IMPORT(C);
    
    vsAdd(row * col, A, B, C);
}

template<int row, int col>
void Plus(Tensor<row, col> _A, const float x, Tensor<row, col> _C) {
    IMPORT_CONST(A);
    IMPORT(C);

    for (int i = 0; i < row * col; i++) {
        C[i] = A[i] + x;
    }
}

template<int row, int col>
void Sub(Tensor<row, col> _A, Tensor<row, col> _B, Tensor<row, col> _C) {
    IMPORT_CONST(A);
    IMPORT_CONST(B);
    IMPORT(C);

    vsSub(row * col, A, B, C);
}

template<int row, int col>
void Sub(Tensor<row, col> _A, const float x, Tensor<row, col> _C) {
    IMPORT_CONST(A);
    IMPORT(C);

    for (int i = 0; i < row * col; i++) {
        C[i] = A[i] - x;
    }
}

template<int row, int col>
void Mul(Tensor<row, col> _A, Tensor<row, col> _B, Tensor<row, col> _C) {
    IMPORT_CONST(A);
    IMPORT_CONST(B);
    IMPORT(C);

    vsMul(row * col, A, B, C);
}

template<int row, int col>
void Mul(Tensor<row, col> _A, const float x, Tensor<row, col> _C) {
    IMPORT_CONST(A);
    IMPORT(C);

    for (int i = 0; i < row * col; i++) {
        C[i] = A[i] * x;
    }
}

template<int row, int col>
void Div(Tensor<row, col> _A, Tensor<row, col> _B, Tensor<row, col> _C) {
    IMPORT_CONST(A);
    IMPORT_CONST(B);
    IMPORT(C);

    vsDiv(row * col, A, B, C);
}

template<int row, int col>
void Div(Tensor<row, col> _A, const float x, Tensor<row, col> _C) {
    IMPORT_CONST(A);
    IMPORT(C);

    const float inv = 1.0f / x;
    for (int i = 0; i < row * col; i++) {
        C[i] = A[i] * inv;
    }
}

template<int row,int col>
void Set(Tensor<row, col> _A, const float x) {
    IMPORT(A);

    std::fill(A, A + row * col, x);
}

template<int row, int col>
void Reset(Tensor<row, col> _A) {
    IMPORT(A);
    
    constexpr int realSize = GetSizeBytes(row, col);
    std::memset(A, 0, realSize);
}

template<int row, int col>
float GetMean(Tensor<row, col> _A) {
    IMPORT(A);

    float mean = 0.0f;
    for(int i = 0;i < row * col;i++) {
        mean += A[i];
    }
    return mean / (row * col);
}

template<int row, int col>
float GetStd(Tensor<row, col> _A, const float mean) {
    IMPORT(A);

    float o2 = 0.0f;
    for(int i = 0;i < row * col;i++) {
        float x = A[i] - mean;
        o2 += x * x;
    }
    return std::sqrt(o2);
}

template<int row,int col>
void Print(Tensor<row, col> _A) {
    IMPORT_CONST(A);

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            std::cout << A[i * col + j] << " ";
        }
        std::cout << std::endl;
    }
}

template<int len>
void ApplyLookAheadMask(Tensor<len, len> _A, int npd, const float x) {
    IMPORT(A);

    for(int j = 0;j < npd;j++) {
        for(int k = j + 1;k < len;k++) {
            A[j*(len) + k] = x;
        }
    }
    for(int j = npd;j<len;j++) {
        for(int k = 0;k < len;k++) {
            A[j*(len) + k] = x;
        }
    }
}

template<int len>
void ApplyPaddingMask(Tensor<len, len> _A, int npd, const float x) {
    IMPORT(A);

    for(int j = 0;j < npd;j++) {
        for(int k = npd;k < len;k++) {
            A[j*(len) + k] = x;
        }
    }
    for(int j = npd;j < len;j++) {
        for(int k = 0;k < len;k++) {
            A[j*(len) + k] = x;
        }
    }
}

template<int len>
void ApplyCrossPaddingMask(Tensor<len, len> _A, int npd, const float x) {
    IMPORT(A);

    for(int j = 0;j < len;j++) {
        for(int k = npd;k < len;k++) {
            A[j*(len) + k] = x;
        }
    }
}

template<int batch,int len, int col>
void GetPositionalEncode(Tensor<batch * len, col> _A) {
    IMPORT(A);

    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < len; j++) {
            for (int k = 0; k < col; k += 2) {
                A[i*(len*col) + j*(col) + k] = std::sin(j / std::pow(10000, float(k) / col));
            }
            for (int k = 1; k < col; k += 2) {
                A[i*(len*col) + j*(col) + k] = std::cos(j / std::pow(10000, float(k - 1) / col));
            }
        }
    }
}


template<int d1, int d2, int d3>
void MatMulPlusATB(Tensor<d2,d1> _A, Tensor<d2,d3> _B, Tensor<d1,d3> _C) {
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, d1, d3, d2, 1.0f, _A.data, d1, _B.data, d3, 1.0f, _C.data, d3);
}

template<int d1,int d2,int d3>
void MatMulPlusABT(Tensor<d1,d2> _A, Tensor<d3,d2> _B, Tensor<d1,d3> _C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, d1, d3, d2, 1.0f, _A.data, d2, _B.data, d2, 1.0f, _C.data, d3);
}

template<int d1, int d2, int d3>
void MatMulPlusAB(Tensor<d1,d2> _A, Tensor<d2,d3> _B, Tensor<d1,d3> _C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, d1, d3, d2, 1.0f, _A.data, d2, _B.data, d3, 1.0f, _C.data, d3);
}


#endif