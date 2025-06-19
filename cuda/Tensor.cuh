#ifndef TENSOR
#define TENSOR

#include "Header.h"
#include "cnpy.h"
#include "UtilKernel.cuh"

class Tensor{
public:
    Tensor() {;}
    Tensor(Tensor& other) : data(other.data), pitch(other.pitch), row(other.row), col(other.col) {;}
    Tensor(const std::size_t row,const std::size_t col) : row(row), col(col) {
        cudaMallocPitch(&data, &pitch, col * sizeof(float), row);
    }

    void free() {
        cudaFree(data);
    }

    void toFloat(float* _data) {
        cudaMemcpy2D(_data, sizeof(float) * col, data, pitch, sizeof(float) * col, row, cudaMemcpyDeviceToHost);
    }

    void loadNp(cnpy::npz_t npFile, std::string name) {
        cnpy::NpyArray arr = npFile[name];
        cudaMemcpy2D(data, pitch, arr.data<float>(), sizeof(float) * col, sizeof(float) * col, row, cudaMemcpyHostToDevice);
    }

    void XavierUniformFill() {
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

    void UniformFill(const float limit) {
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

    void HeNormalFill() {
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
    
    float* data;
    std::size_t pitch;
    std::size_t row;
    std::size_t col;
};

void Copy(Tensor A, Tensor B) {
   cudaMemcpy2D(A.data, A.pitch, B.data, B.pitch, sizeof(float) * A.col, A.row, cudaMemcpyDeviceToDevice);
}

__global__ void PlusKernel(
    const float* A, const float* B, float* C,
    const std::size_t pitchA, const std::size_t pitchB, const std::size_t pitchC,
    const std::size_t row, const std::size_t col) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    if(r < row && c < col) {
        *Get(C, r, c, pitchC) = *Get(A, r, c, pitchA) + *Get(B, r, c, pitchB);
    }
}
void Plus(Tensor A, Tensor B, Tensor C) {
    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(A.col, BLOCKSIZE), ceil(A.row, BLOCKSIZE));
    PlusKernel<<<gridDim, blockDim>>>(A.data, B.data, C.data, A.pitch, B.pitch, C.pitch, A.row, A.col);
}


__global__ void PlusKernel(
    const float* A, const float x, float* C,
    const std::size_t pitchA, const std::size_t pitchC,
    const std::size_t row, const std::size_t col) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    if(r < row && c < col) {
        *Get(C, r, c, pitchC) = *Get(A, r, c, pitchA) + x;
    }
}
void Plus(Tensor A, const float x, Tensor C) {
    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(A.col, BLOCKSIZE), ceil(A.row, BLOCKSIZE));
    PlusKernel<<<gridDim, blockDim>>>(A.data, x, C.data, A.pitch, C.pitch, A.row, A.col);
}


__global__ void SubKernel(
    const float* A, const float* B, float* C,
    const std::size_t pitchA, const std::size_t pitchB, const std::size_t pitchC,
    const std::size_t row, const std::size_t col) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    if(r < row && c < col) {
        *Get(C, r, c, pitchC) = *Get(A, r, c, pitchA) - *Get(B, r, c, pitchB);
    }
}
void Sub(Tensor A, Tensor B, Tensor C) {
    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(A.col, BLOCKSIZE), ceil(A.row, BLOCKSIZE));
    SubKernel<<<gridDim, blockDim>>>(A.data, B.data, C.data, A.pitch, B.pitch, C.pitch, A.row, A.col);
}


__global__ void SubKernel(
    const float* A, const float x, float* C,
    const std::size_t pitchA, const std::size_t pitchC,
    const std::size_t row, const std::size_t col) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    if(r < row && c < col) {
        *Get(C, r, c, pitchC) = *Get(A, r, c, pitchA) - x;
    }
}
void SubKernel(Tensor A, const float x, Tensor C) {
    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(A.col, BLOCKSIZE), ceil(A.row, BLOCKSIZE));
    SubKernel<<<gridDim, blockDim>>>(A.data, x, C.data, A.pitch, C.pitch, A.row, A.col);
}



__global__ void MulKernel(
    const float* A, const float* B, float* C,
    const std::size_t pitchA, const std::size_t pitchB, const std::size_t pitchC,
    const std::size_t row, const std::size_t col) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    if(r < row && c < col) {
        *Get(C, r, c, pitchC) = *Get(A, r, c, pitchA) * *Get(B, r, c, pitchB);
    }
}
void Mul(Tensor A, Tensor B, Tensor C) {
    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(A.col, BLOCKSIZE), ceil(A.row, BLOCKSIZE));
    MulKernel<<<gridDim, blockDim>>>(A.data, B.data, C.data, A.pitch, B.pitch, C.pitch, A.row, A.col);
}


__global__ void MulKernel(
    const float* A, const float x, float* C,
    const std::size_t pitchA, const std::size_t pitchC,
    const std::size_t row, const std::size_t col) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    if(r < row && c < col) {
        *Get(C, r, c, pitchC) = *Get(A, r, c, pitchA) * x;
    }
}
void Mul(Tensor A, const float x, Tensor C) {
    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(A.col, BLOCKSIZE), ceil(A.row, BLOCKSIZE));
    MulKernel<<<gridDim, blockDim>>>(A.data, x, C.data, A.pitch, C.pitch, A.row, A.col);
}



__global__ void DivKernel(
    const float* A, const float* B, float* C,
    const std::size_t pitchA, const std::size_t pitchB, const std::size_t pitchC,
    const std::size_t row, const std::size_t col) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    if(r < row && c < col) {
        *Get(C, r, c, pitchC) = *Get(A, r, c, pitchA) * __frcp_rn(*Get(B, r, c, pitchB));
    }
}
void Div(Tensor A, Tensor B, Tensor C) {
    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(A.col, BLOCKSIZE), ceil(A.row, BLOCKSIZE));
    DivKernel<<<gridDim, blockDim>>>(A.data, B.data, C.data, A.pitch, B.pitch, C.pitch, A.row, A.col);
}


void Div(Tensor A, const float x, Tensor C) {
    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(A.col, BLOCKSIZE), ceil(A.row, BLOCKSIZE));
    MulKernel<<<gridDim, blockDim>>>(A.data, 1.0f / x, C.data, A.pitch, C.pitch, A.row, A.col);
}


__global__ void SetKernel(
    const float* A, const float x,
    const std::size_t pitchA,
    const std::size_t row, const std::size_t col) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    if(r < row && c < col) {
        *Get(A, r, c, pitchA) = x;
    }
}
void Set(Tensor A, const float x) {
    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(A.col, BLOCKSIZE), ceil(A.row, BLOCKSIZE));
    SetKernel<<<gridDim, blockDim>>>(A.data, x, A.pitch, A.row, A.col);
}

void Reset(Tensor A) {
   cudaMemset2D(A.data, A.pitch, 0, sizeof(float) * A.col, A.row);
}

__global__ void ReduceMaxKernel(
    const float* A, const float* C,
    const std::size_t pitchA, const std::size_t pitchC,
    const std::size_t row, const std::size_t col) {
    
}
void ReduceMax(Tensor A, Tensor C) {
    if(A.col <= 1024) {

    }
    else {
        Tensor temp(ceil(A.col, 1024), A.row);

        temp.free();
    }
}



__global__ void ApplyLookAheadMaskBatchKernel(
    const float* A, const int seq, const float x,
    const std::size_t pitchA, 
    const int batch, const std::size_t row, const std::size_t col) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int sr = r % batch;

    if((c > sr || sr >= seq) && (r < row && c < col)) {
        *Get(A, r, c, pitchA) = x;
    }
}
void ApplyLookAheadMaskBatch(Tensor A, const int batch, const int seq, const float x) {
    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(A.col, BLOCKSIZE), ceil(A.row, BLOCKSIZE));
    ApplyLookAheadMaskBatchKernel<<<gridDim, blockDim>>>(A.data, seq, x, A.pitch, batch, A.row, A.col);
//     1 x x x x
//     1 1 x x x
//     1 1 1 x x <- seq
//     x x x x x
//     x x x x x
}


__global__ void ApplyPaddingMaskBatchKernel(
    const float* A, const int seq, const float x,
    const std::size_t pitchA, 
    const int batch, const std::size_t row, const std::size_t col) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int sr = r % batch;

    if((c >= seq || sr >= seq) && (r < row && c < col)) {
        *Get(A, r, c, pitchA) = x;
    }
}
void ApplyPaddingMaskBatch(Tensor A, const int batch, const int seq, const float x) {
    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(A.col, BLOCKSIZE), ceil(A.row, BLOCKSIZE));
    ApplyPaddingMaskBatchKernel<<<gridDim, blockDim>>>(A.data, seq, x, A.pitch, batch, A.row, A.col);
    // 1 1 1 x x x
    // 1 1 1 x x x
    // 1 1 1 x x x <- seq
    // x x x x x x 
    // x x x x x x
}


__global__ void ApplyCrossPaddingMaskBatchKernel(
    const float* A, const int seq, const float x,
    const std::size_t pitchA, 
    const int batch, const std::size_t row, const std::size_t col) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    if(c >= seq && (r < row && c < col)) {
        *Get(A, r, c, pitchA) = x;
    }
}
void ApplyCrossPaddingMaskBatch(Tensor A, const int batch, const int seq, const float x) {
    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(A.col, BLOCKSIZE), ceil(A.row, BLOCKSIZE));
    ApplyCrossPaddingMaskBatchKernel<<<gridDim, blockDim>>>(A.data, seq, x, A.pitch, batch, A.row, A.col);
    // 1 1 1 x x
    // 1 1 1 x x
    // 1 1 1 x x
    // 1 1 1 x x
    // 1 1 1 x x
    //     ^seq
}


// Need -use_fast_math
__global__ void GetPositionalEncodeKernel(
    const float* A, const std::size_t pitchA, 
    const std::size_t row, const std::size_t col) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    if(r < row && c < col) {
        if(c % 2 == 0) {
            *Get(A, r, c, pitchA) = __sinf(r * __frcp_rn(powf(10000.0f, float(c) / col)));
        }
        else {
            *Get(A, r, c, pitchA) = __sinf(r * __frcp_rn(powf(10000.0f, float(c - 1) / col)));
        }
    }
}
void GetPositionalEncode(Tensor A) {
    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(A.col, BLOCKSIZE), ceil(A.row, BLOCKSIZE));
    GetPositionalEncodeKernel<<<gridDim, blockDim>>>(A.data, A.pitch, A.row, A.col);
}


void Print(Tensor A) {
    float* _data = (float*)malloc(sizeof(float) * A.row * A.col);
    cudaMemcpy2D(_data, sizeof(float) * A.col, A.data, A.pitch, sizeof(float) * A.col, A.row, cudaMemcpyDeviceToHost);

    for (int i = 0; i < A.row; i++) {
        for (int j = 0; j < A.col; j++) {
            std::cout << _data[i * A.col + j] << " ";
        }
        std::cout << std::endl;
    }
}


static constexpr int MATMUL_BLOCKSIZE = 16;

// A : d1 * d2
// B : d2 * d3
// C : d1 * d3
__global__ void MatMulKernelAB(
    const float* A, const float* B, const float* C, const std::size_t pitchA, const std::size_t pitchB, const std::size_t pitchC,
    const std::size_t d1, const std::size_t d2, const std::size_t d3) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    float CValue = 0.0f;

    __shared__ float As[MATMUL_BLOCKSIZE][MATMUL_BLOCKSIZE];
    __shared__ float Bs[MATMUL_BLOCKSIZE][MATMUL_BLOCKSIZE];

    if(r < d1 && c < d3) {
        for(int i = 0;i < ceil(d2, MATMUL_BLOCKSIZE);i++) {
           
            float loadIdxX = i * MATMUL_BLOCKSIZE + threadIdx.x;
            float loadIdxY = i * MATMUL_BLOCKSIZE + threadIdx.y;
            if(loadIdxX < d2) {
                As[threadIdx.y][threadIdx.x] = Get(A,r,loadIdxX,pitchA)[0];
            }
            else {
                As[threadIdx.y][threadIdx.x] = 0.0f;
            }
            if(loadIdxY < d2) {
                Bs[threadIdx.y][threadIdx.x] = Get(B,loadIdxY,c,pitchB)[0];
            }
            else {
                Bs[threadIdx.y][threadIdx.x] = 0.0f;
            }
            __syncthreads();
            for(int j = 0;j < MATMUL_BLOCKSIZE;j++) {
                CValue += As[threadIdx.y][j] * Bs[j][threadIdx.x];
            }
            __syncthreads();
        }
        Get(C,r,c,pitchC)[0] = CValue;
    }
}

// A : d2 * d1
// B : d2 * d3
// C : d1 * d3
__global__ void MatMulKernelATB(
    const float* A, const float* B, const float* C, const std::size_t pitchA, const std::size_t pitchB, const std::size_t pitchC,
    const std::size_t d1, const std::size_t d2, const std::size_t d3) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    float CValue = 0.0f;

    __shared__ float As[MATMUL_BLOCKSIZE][MATMUL_BLOCKSIZE];
    __shared__ float Bs[MATMUL_BLOCKSIZE][MATMUL_BLOCKSIZE];

    if(r < d1 && c < d3) {
        for(int i = 0;i < ceil(d2, MATMUL_BLOCKSIZE);i++) {
            float a0 = blockIdx.y * blockDim.y;
            float b0 = blockIdx.x * blockDim.x;
            float loadIdxY = i * MATMUL_BLOCKSIZE + threadIdx.y;
            if(loadIdxY < d2) {
                As[threadIdx.y][threadIdx.x] = Get(A,loadIdxY,a0 + threadIdx.x,pitchA)[0];
                Bs[threadIdx.y][threadIdx.x] = Get(B,loadIdxY,b0 + threadIdx.x,pitchB)[0];
            }
            else {
                As[threadIdx.y][threadIdx.x] = 0.0f;
                Bs[threadIdx.y][threadIdx.x] = 0.0f;
            }
            __syncthreads();
            for(int j = 0;j < MATMUL_BLOCKSIZE;j++) {
                CValue += As[j][threadIdx.y] * Bs[j][threadIdx.x];
            }
            __syncthreads();
        }
        Get(C,r,c,pitchC)[0] = CValue;
    }
}


// A : d1 * d2
// B : d3 * d2
// C : d1 * d3
__global__ void MatMulKernelABT(
    const float* A, const float* B, const float* C, const std::size_t pitchA, const std::size_t pitchB, const std::size_t pitchC,
    const std::size_t d1, const std::size_t d2, const std::size_t d3) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    float CValue = 0.0f;

    __shared__ float As[MATMUL_BLOCKSIZE][MATMUL_BLOCKSIZE];
    __shared__ float Bs[MATMUL_BLOCKSIZE][MATMUL_BLOCKSIZE];

    if(r < d1 && c < d3) {
        for(int i = 0;i < ceil(d2, MATMUL_BLOCKSIZE);i++) {
            float a0 = blockIdx.y * blockDim.y;
            float b0 = blockIdx.x * blockDim.x;
            float loadIdxX = i * MATMUL_BLOCKSIZE + threadIdx.x;
            if(loadIdxX < d2) {
                As[threadIdx.y][threadIdx.x] = Get(A,a0 + threadIdx.y,loadIdxX,pitchA)[0];
                Bs[threadIdx.y][threadIdx.x] = Get(B,b0 + threadIdx.y,loadIdxX,pitchB)[0];
            }
            else {
                As[threadIdx.y][threadIdx.x] = 0.0f;
                Bs[threadIdx.y][threadIdx.x] = 0.0f;
            }
            __syncthreads();
            for(int j = 0;j < MATMUL_BLOCKSIZE;j++) {
                CValue += As[threadIdx.y][j] * Bs[threadIdx.x][j];
            }
            __syncthreads();
        }
        Get(C,r,c,pitchC)[0] = CValue;
    }
}


void MatMulPlusAsync(const Tensor a,const Tensor b, const Tensor c, bool ATransposed, bool BTransposed) {
    if(!ATransposed && !BTransposed) {
        dim3 blockDim(MATMUL_BLOCKSIZE, MATMUL_BLOCKSIZE);
        dim3 gridDim(ceil(c.col, MATMUL_BLOCKSIZE), ceil(c.row, MATMUL_BLOCKSIZE));
        MatMulKernelAB<<<gridDim, blockDim>>>(a.data, b.data, c.data, a.pitch, b.pitch, c.pitch, c.row, a.col, b.col);
    }
    else if(ATransposed && !BTransposed) {
        dim3 blockDim(MATMUL_BLOCKSIZE, MATMUL_BLOCKSIZE);
        dim3 gridDim(ceil(c.col, MATMUL_BLOCKSIZE), ceil(c.row, MATMUL_BLOCKSIZE));
        MatMulKernelATB<<<gridDim, blockDim>>>(a.data, b.data, c.data, a.pitch, b.pitch, c.pitch, c.row, a.row, c.col);
    }
    else if (!ATransposed && BTransposed) {
        dim3 blockDim(MATMUL_BLOCKSIZE, MATMUL_BLOCKSIZE);
        dim3 gridDim(ceil(c.col, MATMUL_BLOCKSIZE), ceil(c.row, MATMUL_BLOCKSIZE));
        MatMulKernelABT<<<gridDim, blockDim>>>(a.data, b.data, c.data, a.pitch, b.pitch, c.pitch, c.row, a.col, c.col);
    }
    else {
        // nothing implemented here.
    }
}

#endif