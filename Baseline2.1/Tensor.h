#ifndef TENSOR
#define TENSOR

#include "Header.h"
#include "cnpy.h"

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

    void loadNp(cnpy::npz_t npFile, std::string name) {
        cnpy::NpyArray arr = npFile[name];
        std::memcpy(data, arr.data<float>(), sizeof(float) * row * col);
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
    
    for (int i = 0; i < row * col; i++) {
        C[i] = A[i] + B[i];
    }
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

    for (int i = 0; i < row * col; i++) {
        C[i] = A[i] - B[i];
    }
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

    for (int i = 0; i < row * col; i++) {
        C[i] = A[i] * B[i];
    }
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

    for (int i = 0; i < row * col; i++) {
        C[i] = A[i] / B[i];
    }
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

template<int batch,int len>
void ApplyLookAheadMask(Tensor<batch * len, len> _A, int npd, const float x) {
    IMPORT(A);

    for(int i = 0 ;i < batch;i++) {
        for(int j = 0;j < npd;j++) {
            for(int k = j + 1;k < len;k++) {
                A[i*(len*len) + j*(len) + k] = x;
            }
        }
        for(int j = npd;j<len;j++) {
            for(int k = 0;k < len;k++) {
                A[i*(len*len) + j*(len) + k] = x;
            }
        }
    }
}

template<int batch,int len>
void ApplyPaddingMask(Tensor<batch * len, len> _A, int npd, const float x) {
    IMPORT(A);

    for(int i = 0 ;i < batch;i++) {
        for(int j = 0;j < npd;j++) {
            for(int k = npd;k < len;k++) {
                A[i*(len*len) + j*(len) + k] = x;
            }
        }
        for(int j = npd;j < len;j++) {
            for(int k = 0;k < len;k++) {
                A[i*(len*len) + j*(len) + k] = x;
            }
        }
    }
}

template<int batch, int len>
void ApplyCrossPaddingMask(Tensor<batch * len, len> _A, int npd, const float x) {
    IMPORT(A);

    for(int i = 0;i < batch;i++) {
        for(int j = 0;j < len;j++) {
            for(int k = npd;k < len;k++) {
                A[i*(len*len) + j*(len) + k] = x;
            }
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
    IMPORT_CONST(A);
    IMPORT_CONST(B);
    IMPORT(C);

    if constexpr (d3 * d1 >= 8 * 1024) {
        constexpr int BLOCK_SIZE1 = 8;
        constexpr int BLOCK_SIZE3 = 512;

        constexpr int _ii = d1 / BLOCK_SIZE1 * BLOCK_SIZE1;
        constexpr int _jj = d3 / BLOCK_SIZE3 * BLOCK_SIZE3;
        for (int ii = 0; ii < _ii; ii += BLOCK_SIZE1) {
            for (int jj = 0; jj < _jj; jj += BLOCK_SIZE3) {
                for (int k = 0; k < d2; k++) {
                    for (int i = 0; i < BLOCK_SIZE1; i++) {
                        for (int j = 0; j < BLOCK_SIZE3; j++) {
                            C[(ii + i) * d3 + (jj + j)] += A[k * d1 + (ii + i)] * B[k * d3 + (jj + j)];
                        }
                    }
                }
            }
            for (int k = 0; k < d2; k++) {
                for (int i = 0; i < BLOCK_SIZE1; i++) {
                    for (int j = 0; j < d3 % BLOCK_SIZE3; j++) {
                        C[(ii + i) * d3 + (_jj + j)] += A[k * d1 + (ii + i)] * B[k * d3 + (_jj + j)];
                    }
                }
            }
        }
        for (int jj = 0; jj < _jj; jj += BLOCK_SIZE3) {
            for (int k = 0; k < d2; k++) {
                for (int i = 0; i < d1 % BLOCK_SIZE1; i++) {
                    for (int j = 0; j < BLOCK_SIZE3; j++) {
                        C[(_ii + i) * d3 + (jj + j)] += A[k * d1 + (_ii + i)] * B[k * d3 + (jj + j)];
                    }
                }
            }
        }
        for (int k = 0; k < d2; k++) {
            for (int i = 0; i < d1 % BLOCK_SIZE1; i++) {
                for (int j = 0; j < d3 % BLOCK_SIZE3; j++) {
                    C[(_ii + i) * d3 + (_jj + j)] += A[k * d1 + (_ii + i)] * B[k * d3 + (_jj + j)];
                }
            }
        }
    }
    else {
        constexpr int BLOCK_SIZE = 80;

        constexpr int _ii = d1 / BLOCK_SIZE * BLOCK_SIZE;
        constexpr int _jj = d3 / BLOCK_SIZE * BLOCK_SIZE;
        for (int ii = 0; ii < _ii; ii += BLOCK_SIZE) {
            for (int jj = 0; jj < _jj; jj += BLOCK_SIZE) {
                for (int k = 0; k < d2; k++) {
                    for (int i = 0; i < BLOCK_SIZE; i++) {
                        for (int j = 0; j < BLOCK_SIZE; j++) {
                            C[(ii + i) * d3 + (jj + j)] += A[k * d1 + (ii + i)] * B[k * d3 + (jj + j)];
                        }
                    }
                }
            }
            for (int k = 0; k < d2; k++) {
                for (int i = 0; i < BLOCK_SIZE; i++) {
                    for (int j = 0; j < d3 % BLOCK_SIZE; j++) {
                        C[(ii + i) * d3 + (_jj + j)] += A[k * d1 + (ii + i)] * B[k * d3 + (_jj + j)];
                    }
                }
            }
        }
        for (int jj = 0; jj < _jj; jj += BLOCK_SIZE) {
            for (int k = 0; k < d2; k++) {
                for (int i = 0; i < d1 % BLOCK_SIZE; i++) {
                    for (int j = 0; j < BLOCK_SIZE; j++) {
                        C[(_ii + i) * d3 + (jj + j)] += A[k * d1 + (_ii + i)] * B[k * d3 + (jj + j)];
                    }
                }
            }
        }
        for (int k = 0; k < d2; k++) {
            for (int i = 0; i < d1 % BLOCK_SIZE; i++) {
                for (int j = 0; j < d3 % BLOCK_SIZE; j++) {
                    C[(_ii + i) * d3 + (_jj + j)] += A[k * d1 + (_ii + i)] * B[k * d3 + (_jj + j)];
                }
            }
        }
    }
}

template<int d1,int d2,int d3>
void MatMulPlusABT(Tensor<d1,d2> _A, Tensor<d3,d2> _B, Tensor<d1,d3> _C) {
    IMPORT_CONST(A);
    IMPORT_CONST(B);
    IMPORT(C);

    if constexpr (d3 * d1 >= 8 * 1024) {
        constexpr int BLOCK_SIZE1 = 8;
        constexpr int BLOCK_SIZE3 = 512;

        constexpr int _ii = d1 / BLOCK_SIZE1 * BLOCK_SIZE1;
        constexpr int _jj = d3 / BLOCK_SIZE3 * BLOCK_SIZE3;
        for (int ii = 0; ii < _ii; ii += BLOCK_SIZE1) {
            for (int jj = 0; jj < _jj; jj += BLOCK_SIZE3) {
                for (int k = 0; k < d2; k++) {
                    for (int i = 0; i < BLOCK_SIZE1; i++) {
                        for (int j = 0; j < BLOCK_SIZE3; j++) {
                            C[(ii + i) * d3 + (jj + j)] += A[(ii + i) * d2 + k] * B[(jj + j) * d2 + k];
                        }
                    }
                }
            }
            for (int k = 0; k < d2; k++) {
                for (int i = 0; i < BLOCK_SIZE1; i++) {
                    for (int j = 0; j < d3 % BLOCK_SIZE3; j++) {
                        C[(ii + i) * d3 + (_jj + j)] += A[(ii + i) * d2 + k] * B[(_jj + j) * d2 + k];
                    }
                }
            }
        }
        for (int jj = 0; jj < _jj; jj += BLOCK_SIZE3) {
            for (int k = 0; k < d2; k++) {
                for (int i = 0; i < d1 % BLOCK_SIZE1; i++) {
                    for (int j = 0; j < BLOCK_SIZE3; j++) {
                        C[(_ii + i) * d3 + (jj + j)] += A[(_ii + i) * d2 + k] * B[(jj + j) * d2 + k];
                    }
                }
            }
        }
        for (int k = 0; k < d2; k++) {
            for (int i = 0; i < d1 % BLOCK_SIZE1; i++) {
                for (int j = 0; j < d3 % BLOCK_SIZE3; j++) {
                    C[(_ii + i) * d3 + (_jj + j)] += A[(_ii + i) * d2 + k] * B[(_jj + j) * d2 + k];
                }
            }
        }
    }
    else {
        constexpr int BLOCK_SIZE = 80;

        constexpr int _ii = d1 / BLOCK_SIZE * BLOCK_SIZE;
        constexpr int _jj = d3 / BLOCK_SIZE * BLOCK_SIZE;
        for (int ii = 0; ii < _ii; ii += BLOCK_SIZE) {
            for (int jj = 0; jj < _jj; jj += BLOCK_SIZE) {
                for (int k = 0; k < d2; k++) {
                    for (int i = 0; i < BLOCK_SIZE; i++) {
                        for (int j = 0; j < BLOCK_SIZE; j++) {
                            C[(ii + i) * d3 + (jj + j)] += A[(ii + i) * d2 + k] * B[(jj + j) * d2 + k];
                        }
                    }
                }
            }
            for (int k = 0; k < d2; k++) {
                for (int i = 0; i < BLOCK_SIZE; i++) {
                    for (int j = 0; j < d3 % BLOCK_SIZE; j++) {
                        C[(ii + i) * d3 + (_jj + j)] += A[(ii + i) * d2 + k] * B[(_jj + j) * d2 + k];
                    }
                }
            }
        }
        for (int jj = 0; jj < _jj; jj += BLOCK_SIZE) {
            for (int k = 0; k < d2; k++) {
                for (int i = 0; i < d1 % BLOCK_SIZE; i++) {
                    for (int j = 0; j < BLOCK_SIZE; j++) {
                        C[(_ii + i) * d3 + (jj + j)] += A[(_ii + i) * d2 + k] * B[(jj + j) * d2 + k];
                    }
                }
            }
        }
        for (int k = 0; k < d2; k++) {
            for (int i = 0; i < d1 % BLOCK_SIZE; i++) {
                for (int j = 0; j < d3 % BLOCK_SIZE; j++) {
                    C[(_ii + i) * d3 + (_jj + j)] += A[(_ii + i) * d2 + k] * B[(_jj + j) * d2 + k];
                }
            }
        }
    }
}

template<int d1, int d2, int d3>
void MatMulPlusAB(Tensor<d1,d2> _A, Tensor<d2,d3> _B, Tensor<d1,d3> _C) {
    IMPORT_CONST(A);
    IMPORT_CONST(B);
    IMPORT(C);

    if constexpr (d3 * d1 >= 8 * 1024) {
        constexpr int BLOCK_SIZE1 = 8;
        constexpr int BLOCK_SIZE3 = 512;

        constexpr int _ii = d1 / BLOCK_SIZE1 * BLOCK_SIZE1;
        constexpr int _jj = d3 / BLOCK_SIZE3 * BLOCK_SIZE3;
        for (int ii = 0; ii < _ii; ii += BLOCK_SIZE1) {
            for (int jj = 0; jj < _jj; jj += BLOCK_SIZE3) {
                for (int k = 0; k < d2; k++) {
                    for (int i = 0; i < BLOCK_SIZE1; i++) {
                        for (int j = 0; j < BLOCK_SIZE3; j++) {
                            C[(ii + i) * d3 + (jj + j)] += A[(ii + i) * d2 + k] * B[k * d3 + (jj + j)];
                        }
                    }
                }
            }
            for (int k = 0; k < d2; k++) {
                for (int i = 0; i < BLOCK_SIZE1; i++) {
                    for (int j = 0; j < d3 % BLOCK_SIZE3; j++) {
                        C[(ii + i) * d3 + (_jj + j)] += A[(ii + i) * d2 + k] * B[k * d3 + (_jj + j)];
                    }
                }
            }
        }
        for (int jj = 0; jj < _jj; jj += BLOCK_SIZE3) {
            for (int k = 0; k < d2; k++) {
                for (int i = 0; i < d1 % BLOCK_SIZE1; i++) {
                    for (int j = 0; j < BLOCK_SIZE3; j++) {
                        C[(_ii + i) * d3 + (jj + j)] += A[(_ii + i) * d2 + k] * B[k * d3 + (jj + j)];
                    }
                }
            }
        }
        for (int k = 0; k < d2; k++) {
            for (int i = 0; i < d1 % BLOCK_SIZE1; i++) {
                for (int j = 0; j < d3 % BLOCK_SIZE3; j++) {
                    C[(_ii + i) * d3 + (_jj + j)] += A[(_ii + i) * d2 + k] * B[k * d3 + (_jj + j)];
                }
            }
        }
    }
    else {
        constexpr int BLOCK_SIZE = 80;

        constexpr int _ii = d1 / BLOCK_SIZE * BLOCK_SIZE;
        constexpr int _jj = d3 / BLOCK_SIZE * BLOCK_SIZE;
        for (int ii = 0; ii < _ii; ii += BLOCK_SIZE) {
            for (int jj = 0; jj < _jj; jj += BLOCK_SIZE) {
                for (int k = 0; k < d2; k++) {
                    for (int i = 0; i < BLOCK_SIZE; i++) {
                        for (int j = 0; j < BLOCK_SIZE; j++) {
                            C[(ii + i) * d3 + (jj + j)] += A[(ii + i) * d2 + k] * B[k * d3 + (jj + j)];
                        }
                    }
                }
            }
            for (int k = 0; k < d2; k++) {
                for (int i = 0; i < BLOCK_SIZE; i++) {
                    for (int j = 0; j < d3 % BLOCK_SIZE; j++) {
                        C[(ii + i) * d3 + (_jj + j)] += A[(ii + i) * d2 + k] * B[k * d3 + (_jj + j)];
                    }
                }
            }
        }
        for (int jj = 0; jj < _jj; jj += BLOCK_SIZE) {
            for (int k = 0; k < d2; k++) {
                for (int i = 0; i < d1 % BLOCK_SIZE; i++) {
                    for (int j = 0; j < BLOCK_SIZE; j++) {
                        C[(_ii + i) * d3 + (jj + j)] += A[(_ii + i) * d2 + k] * B[k * d3 + (jj + j)];
                    }
                }
            }
        }
        for (int k = 0; k < d2; k++) {
            for (int i = 0; i < d1 % BLOCK_SIZE; i++) {
                for (int j = 0; j < d3 % BLOCK_SIZE; j++) {
                    C[(_ii + i) * d3 + (_jj + j)] += A[(_ii + i) * d2 + k] * B[k * d3 + (_jj + j)];
                }
            }
        }
    }
}


#endif