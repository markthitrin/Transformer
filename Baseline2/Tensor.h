#ifndef TENSOR
#define TENSOR

using Tensor = float*;

constexpr int GetColSizeFloat(int col) {
    if (col % 8 == 0) {
        return col;
    }
    else {
        return ((col / 8 + 1) * 8);
    }
}

constexpr int GetColSizeBytes(int col) {
    return GetColSizeFloat(col) * 4;
}

constexpr int GetSizeBytes(int d, int col) {
    return d * GetColSizeBytes(col);
}


template<int d, int col>
Tensor Create() {
    constexpr int realSize = GetSizeBytes(d, col);
    void* data = _aligned_malloc(realSize, 32);
    return (float*)data;
}

template<int d, int col>
Tensor Create0() {
    constexpr int realSize = GetSizeBytes(d, col);
    void* data = _aligned_malloc(realSize, 32);
    std::memset(data, 0, realSize);
    return (float*)data;
}

template<int d, int col>
void FromArray(float* f, Tensor _A) {
    char* itF = reinterpret_cast<char*>(f);
    char* data = reinterpret_cast<char*>(_A);
    constexpr int colSizeBytes = GetColSizeBytes(col);
    constexpr int colDSizeBytes = col * sizeof(float);
    for (int i = 0; i < d; i++) {
        std::memcpy(data, itF, colDSizeBytes);
        data += colSizeBytes;
        itF += colDSizeBytes;
    }
}

template<int d, int col>
void ToArray(float* out, Tensor _A) {
    char* itF = reinterpret_cast<char*>(out);
    char* data = reinterpret_cast<char*>(_A);
    constexpr int colSizeBytes = GetColSizeBytes(col);
    constexpr int colDSizeBytes = col * sizeof(float);
    for (int i = 0; i < d; i++) {
        std::memcpy(itF, data, colDSizeBytes);
        data += colSizeBytes;
        itF += colDSizeBytes;
    }
}

template<int d,int col>
void Copy(Tensor _A, Tensor _B) {
    IMPORT_CONST(A);
    IMPORT(B);
    constexpr int realSize = GetSizeBytes(d, col);
    std::memcpy((void*)B, (void*)A, realSize);
}

template<int d, int col>
void Plus(Tensor _A, Tensor _B, Tensor _C) {
    IMPORT_CONST(A);
    IMPORT_CONST(B);
    IMPORT(C);
    constexpr int _col = GetColSizeFloat(col);
    for (int i = 0; i < d * _col; i++) {
        C[i] = A[i] + B[i];
    }
}

template<int d, int col>
void Plus(Tensor _A, const float x, Tensor& _C) {
    IMPORT_CONST(A);
    IMPORT(C);
    constexpr int _col = GetColSizeFloat(col);
    for (int i = 0; i < d * _col; i++) {
        C[i] = A[i] + x;
    }
}

template<int d, int col>
void Sub(Tensor _A, Tensor _B, Tensor _C) {
    IMPORT_CONST(A);
    IMPORT_CONST(B);
    IMPORT(C);
    constexpr int _col = GetColSizeFloat(col);
    for (int i = 0; i < d * _col; i++) {
        C[i] = A[i] - B[i];
    }
}

template<int d, int col>
void Sub(Tensor _A, const float x, Tensor _C) {
    IMPORT_CONST(A);
    IMPORT(C);
    constexpr int _col = GetColSizeFloat(col);
    for (int i = 0; i < d * _col; i++) {
        C[i] = A[i] - x;
    }
}

template<int d, int col>
void Mul(Tensor _A, Tensor _B, Tensor _C) {
    IMPORT_CONST(A);
    IMPORT_CONST(B);
    IMPORT(C);
    constexpr int _col = GetColSizeFloat(col);
    for (int i = 0; i < d * _col; i++) {
        C[i] = A[i] * B[i];
    }
}

template<int d, int col>
void Mul(Tensor _A, const float x, Tensor _C) {
    IMPORT_CONST(A);
    IMPORT(C);
    constexpr int _col = GetColSizeFloat(col);
    for (int i = 0; i < d * _col; i++) {
        C[i] = A[i] * x;
    }
}

template<int d, int col>
void Div(Tensor _A, Tensor _B, Tensor _C) {
    IMPORT_CONST(A);
    IMPORT_CONST(B);
    IMPORT(C);
    constexpr int _col = GetColSizeFloat(col);
    for (int i = 0; i < d * _col; i++) {
        C[i] = A[i] / B[i];
    }
}

template<int d, int col>
void Div(Tensor _A, const float x, Tensor _C) {
    IMPORT_CONST(A);
    IMPORT(C);
    constexpr int _col = GetColSizeFloat(col);
    const float inv = 1.0f / x;
    for (int i = 0; i < d * _col; i++) {
        C[i] = A[i] * inv;
    }
}

template<int d,int col>
void Set(Tensor _A, const float x) {
    IMPORT(A);
    constexpr int _col = GetColSizeFloat(col);
    std::fill(A, A + d * _col, x);
}

template<int d, int col>
void Reset(Tensor _A) {
    IMPORT(A);
    constexpr int realSize = GetSizeBytes(d, col);
    std::memset(A, 0, realSize);
}

template<int d,int col>
void Print(Tensor _A) {
    IMPORT_CONST(A);
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < col; j++) {
            std::cout << A[i * col + j] << " ";
        }
        std::cout << std::endl;
    }
}

template<int d,int l, float x>
void ApplyLookAheadMask(Tensor _A) {
    IMPORT(A);
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < l; j++) {
            for (int k = j + 1; k < l; k++) {
                A[i * (l * l) + (j * l + k)] = x;
            }
        }
    }
}

template<int d,int l,int col>
void GetPositionalEncode(Tensor _A) {
    IMPORT(A);
    constexpr int _col = GetColSizeFloat(col);
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < l; j++) {
            const int ij = (i * l + j);
            for (int k = 0; k < _col; k += 2) {
                A[ij * _col + k] = std::sin(j / std::pow(10000, float(k) / _col));
            }
            for (int k = 1; k < _col; k += 2) {
                A[ij * _col + k] = std::cos(j / std::pow(10000, float(k - 1) / _col));
            }
        }
    }
}


template<int d1, int d2, int d3>
void MatMulPlusATB(Tensor _A, Tensor _B, Tensor _C) {
    IMPORT_CONST(A);
    IMPORT_CONST(B);
    IMPORT(C);
    constexpr int _d1 = GetColSizeFloat(d1);
    constexpr int _d3 = GetColSizeFloat(d3);

    if constexpr (_d3 * _d1 >= 8 * 1024) {
        constexpr int BLOCK_SIZE1 = 8;
        constexpr int BLOCK_SIZE3 = 512;

        constexpr int _ii = _d1 / BLOCK_SIZE1 * BLOCK_SIZE1;
        constexpr int _jj = _d3 / BLOCK_SIZE3 * BLOCK_SIZE3;
        for (int ii = 0; ii < _ii; ii += BLOCK_SIZE1) {
            for (int jj = 0; jj < _jj; jj += BLOCK_SIZE3) {
                for (int k = 0; k < d2; k++) {
                    for (int i = 0; i < BLOCK_SIZE1; i++) {
                        for (int j = 0; j < BLOCK_SIZE3; j++) {
                            C[(ii + i) * _d3 + (jj + j)] += A[k * _d1 + (ii + i)] * B[k * _d3 + (jj + j)];
                        }
                    }
                }
            }
            for (int k = 0; k < d2; k++) {
                for (int i = 0; i < BLOCK_SIZE1; i++) {
                    for (int j = 0; j < _d3 % BLOCK_SIZE3; j++) {
                        C[(ii + i) * _d3 + (_jj + j)] += A[k * _d1 + (ii + i)] * B[k * _d3 + (_jj + j)];
                    }
                }
            }
        }
        for (int jj = 0; jj < _jj; jj += BLOCK_SIZE3) {
            for (int k = 0; k < d2; k++) {
                for (int i = 0; i < _d1 % BLOCK_SIZE1; i++) {
                    for (int j = 0; j < BLOCK_SIZE3; j++) {
                        C[(_ii + i) * _d3 + (jj + j)] += A[k * _d1 + (_ii + i)] * B[k * _d3 + (jj + j)];
                    }
                }
            }
        }
        for (int k = 0; k < d2; k++) {
            for (int i = 0; i < _d1 % BLOCK_SIZE1; i++) {
                for (int j = 0; j < _d3 % BLOCK_SIZE3; j++) {
                    C[(_ii + i) * _d3 + (_jj + j)] += A[k * _d1 + (_ii + i)] * B[k * _d3 + (_jj + j)];
                }
            }
        }
    }
    else {
        constexpr int BLOCK_SIZE = 80;

        constexpr int _ii = _d1 / BLOCK_SIZE * BLOCK_SIZE;
        constexpr int _jj = _d3 / BLOCK_SIZE * BLOCK_SIZE;
        for (int ii = 0; ii < _ii; ii += BLOCK_SIZE) {
            for (int jj = 0; jj < _jj; jj += BLOCK_SIZE) {
                for (int k = 0; k < d2; k++) {
                    for (int i = 0; i < BLOCK_SIZE; i++) {
                        for (int j = 0; j < BLOCK_SIZE; j++) {
                            C[(ii + i) * _d3 + (jj + j)] += A[k * _d1 + (ii + i)] * B[k * _d3 + (jj + j)];
                        }
                    }
                }
            }
            for (int k = 0; k < d2; k++) {
                for (int i = 0; i < BLOCK_SIZE; i++) {
                    for (int j = 0; j < _d3 % BLOCK_SIZE; j++) {
                        C[(ii + i) * _d3 + (_jj + j)] += A[k * _d1 + (ii + i)] * B[k * _d3 + (_jj + j)];
                    }
                }
            }
        }
        for (int jj = 0; jj < _jj; jj += BLOCK_SIZE) {
            for (int k = 0; k < d2; k++) {
                for (int i = 0; i < _d1 % BLOCK_SIZE; i++) {
                    for (int j = 0; j < BLOCK_SIZE; j++) {
                        C[(_ii + i) * _d3 + (jj + j)] += A[k * _d1 + (_ii + i)] * B[k * _d3 + (jj + j)];
                    }
                }
            }
        }
        for (int k = 0; k < d2; k++) {
            for (int i = 0; i < _d1 % BLOCK_SIZE; i++) {
                for (int j = 0; j < _d3 % BLOCK_SIZE; j++) {
                    C[(_ii + i) * _d3 + (_jj + j)] += A[k * _d1 + (_ii + i)] * B[k * _d3 + (_jj + j)];
                }
            }
        }
    }
}

template<int d1,int d2,int d3>
void MatMulPlusABT(Tensor _A, Tensor _B, Tensor _C) {
    IMPORT_CONST(A);
    IMPORT_CONST(B);
    IMPORT(C);
    constexpr int _d2 = GetColSizeFloat(d2);
    constexpr int _d3 = GetColSizeFloat(d3);

    if constexpr (_d3 * d1 >= 8 * 1024) {
        constexpr int BLOCK_SIZE1 = 8;
        constexpr int BLOCK_SIZE3 = 512;

        constexpr int _ii = d1 / BLOCK_SIZE1 * BLOCK_SIZE1;
        constexpr int _jj = _d3 / BLOCK_SIZE3 * BLOCK_SIZE3;
        for (int ii = 0; ii < _ii; ii += BLOCK_SIZE1) {
            for (int jj = 0; jj < _jj; jj += BLOCK_SIZE3) {
                for (int k = 0; k < _d2; k++) {
                    for (int i = 0; i < BLOCK_SIZE1; i++) {
                        for (int j = 0; j < BLOCK_SIZE3; j++) {
                            C[(ii + i) * _d3 + (jj + j)] += A[(ii + i) * _d2 + k] * B[(jj + j) * _d2 + k];
                        }
                    }
                }
            }
            for (int k = 0; k < _d2; k++) {
                for (int i = 0; i < BLOCK_SIZE1; i++) {
                    for (int j = 0; j < _d3 % BLOCK_SIZE3; j++) {
                        C[(ii + i) * _d3 + (_jj + j)] += A[(ii + i) * _d2 + k] * B[(_jj + j) * _d2 + k];
                    }
                }
            }
        }
        for (int jj = 0; jj < _jj; jj += BLOCK_SIZE3) {
            for (int k = 0; k < _d2; k++) {
                for (int i = 0; i < d1 % BLOCK_SIZE1; i++) {
                    for (int j = 0; j < BLOCK_SIZE3; j++) {
                        C[(_ii + i) * _d3 + (jj + j)] += A[(_ii + i) * _d2 + k] * B[(jj + j) * _d2 + k];
                    }
                }
            }
        }
        for (int k = 0; k < _d2; k++) {
            for (int i = 0; i < d1 % BLOCK_SIZE1; i++) {
                for (int j = 0; j < _d3 % BLOCK_SIZE3; j++) {
                    C[(_ii + i) * _d3 + (_jj + j)] += A[(_ii + i) * _d2 + k] * B[(_jj + j) * _d2 + k];
                }
            }
        }
    }
    else {
        constexpr int BLOCK_SIZE = 80;

        constexpr int _ii = d1 / BLOCK_SIZE * BLOCK_SIZE;
        constexpr int _jj = _d3 / BLOCK_SIZE * BLOCK_SIZE;
        for (int ii = 0; ii < _ii; ii += BLOCK_SIZE) {
            for (int jj = 0; jj < _jj; jj += BLOCK_SIZE) {
                for (int k = 0; k < _d2; k++) {
                    for (int i = 0; i < BLOCK_SIZE; i++) {
                        for (int j = 0; j < BLOCK_SIZE; j++) {
                            C[(ii + i) * _d3 + (jj + j)] += A[(ii + i) * _d2 + k] * B[(jj + j) * _d2 + k];
                        }
                    }
                }
            }
            for (int k = 0; k < _d2; k++) {
                for (int i = 0; i < BLOCK_SIZE; i++) {
                    for (int j = 0; j < _d3 % BLOCK_SIZE; j++) {
                        C[(ii + i) * _d3 + (_jj + j)] += A[(ii + i) * _d2 + k] * B[(_jj + j) * _d2 + k];
                    }
                }
            }
        }
        for (int jj = 0; jj < _jj; jj += BLOCK_SIZE) {
            for (int k = 0; k < _d2; k++) {
                for (int i = 0; i < d1 % BLOCK_SIZE; i++) {
                    for (int j = 0; j < BLOCK_SIZE; j++) {
                        C[(_ii + i) * _d3 + (jj + j)] += A[(_ii + i) * _d2 + k] * B[(jj + j) * _d2 + k];
                    }
                }
            }
        }
        for (int k = 0; k < _d2; k++) {
            for (int i = 0; i < d1 % BLOCK_SIZE; i++) {
                for (int j = 0; j < _d3 % BLOCK_SIZE; j++) {
                    C[(_ii + i) * _d3 + (_jj + j)] += A[(_ii + i) * _d2 + k] * B[(_jj + j) * _d2 + k];
                }
            }
        }
    }
}

template<int d1, int d2, int d3>
void MatMulPlusAB(Tensor _A, Tensor _B, Tensor _C) {
    IMPORT_CONST(A);
    IMPORT_CONST(B);
    IMPORT(C);
    constexpr int _d2 = GetColSizeFloat(d2);
    constexpr int _d3 = GetColSizeFloat(d3);

    if constexpr (_d3 * d1 >= 8 * 1024) {
        constexpr int BLOCK_SIZE1 = 8;
        constexpr int BLOCK_SIZE3 = 512;

        constexpr int _ii = d1 / BLOCK_SIZE1 * BLOCK_SIZE1;
        constexpr int _jj = _d3 / BLOCK_SIZE3 * BLOCK_SIZE3;
        for (int ii = 0; ii < _ii; ii += BLOCK_SIZE1) {
            for (int jj = 0; jj < _jj; jj += BLOCK_SIZE3) {
                for (int k = 0; k < _d2; k++) {
                    for (int i = 0; i < BLOCK_SIZE1; i++) {
                        for (int j = 0; j < BLOCK_SIZE3; j++) {
                            C[(ii + i) * _d3 + (jj + j)] += A[(ii + i) * _d2 + k] * B[k * _d3 + (jj + j)];
                        }
                    }
                }
            }
            for (int k = 0; k < _d2; k++) {
                for (int i = 0; i < BLOCK_SIZE1; i++) {
                    for (int j = 0; j < _d3 % BLOCK_SIZE3; j++) {
                        C[(ii + i) * _d3 + (_jj + j)] += A[(ii + i) * _d2 + k] * B[k * _d3 + (_jj + j)];
                    }
                }
            }
        }
        for (int jj = 0; jj < _jj; jj += BLOCK_SIZE3) {
            for (int k = 0; k < _d2; k++) {
                for (int i = 0; i < d1 % BLOCK_SIZE1; i++) {
                    for (int j = 0; j < BLOCK_SIZE3; j++) {
                        C[(_ii + i) * _d3 + (jj + j)] += A[(_ii + i) * _d2 + k] * B[k * _d3 + (jj + j)];
                    }
                }
            }
        }
        for (int k = 0; k < _d2; k++) {
            for (int i = 0; i < d1 % BLOCK_SIZE1; i++) {
                for (int j = 0; j < _d3 % BLOCK_SIZE3; j++) {
                    C[(_ii + i) * _d3 + (_jj + j)] += A[(_ii + i) * _d2 + k] * B[k * _d3 + (_jj + j)];
                }
            }
        }
    }
    else {
        constexpr int BLOCK_SIZE = 80;

        constexpr int _ii = d1 / BLOCK_SIZE * BLOCK_SIZE;
        constexpr int _jj = _d3 / BLOCK_SIZE * BLOCK_SIZE;
        for (int ii = 0; ii < _ii; ii += BLOCK_SIZE) {
            for (int jj = 0; jj < _jj; jj += BLOCK_SIZE) {
                for (int k = 0; k < _d2; k++) {
                    for (int i = 0; i < BLOCK_SIZE; i++) {
                        for (int j = 0; j < BLOCK_SIZE; j++) {
                            C[(ii + i) * _d3 + (jj + j)] += A[(ii + i) * _d2 + k] * B[k * _d3 + (jj + j)];
                        }
                    }
                }
            }
            for (int k = 0; k < _d2; k++) {
                for (int i = 0; i < BLOCK_SIZE; i++) {
                    for (int j = 0; j < _d3 % BLOCK_SIZE; j++) {
                        C[(ii + i) * _d3 + (_jj + j)] += A[(ii + i) * _d2 + k] * B[k * _d3 + (_jj + j)];
                    }
                }
            }
        }
        for (int jj = 0; jj < _jj; jj += BLOCK_SIZE) {
            for (int k = 0; k < _d2; k++) {
                for (int i = 0; i < d1 % BLOCK_SIZE; i++) {
                    for (int j = 0; j < BLOCK_SIZE; j++) {
                        C[(_ii + i) * _d3 + (jj + j)] += A[(_ii + i) * _d2 + k] * B[k * _d3 + (jj + j)];
                    }
                }
            }
        }
        for (int k = 0; k < _d2; k++) {
            for (int i = 0; i < d1 % BLOCK_SIZE; i++) {
                for (int j = 0; j < _d3 % BLOCK_SIZE; j++) {
                    C[(_ii + i) * _d3 + (_jj + j)] += A[(_ii + i) * _d2 + k] * B[k * _d3 + (_jj + j)];
                }
            }
        }
    }
}


#endif