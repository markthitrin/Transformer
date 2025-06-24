#include "Header.cuh"
#include "Tensor.cuh"
#include "UtilKernel.cuh"

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
__global__ void PlusInplaceKernel(
    float* A, const float* B,
    const std::size_t pitchA, const std::size_t pitchB,
    const std::size_t row, const std::size_t col) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    if(r < row && c < col) {
        *Get(A, r, c, pitchA) += *Get(B, r, c, pitchB);
    }
}
__global__ void PlusBatchKernel(
    const float* A, const float* B, float* C,
    const std::size_t pitchA,const std::size_t pitchB, const std::size_t pitchC,
    const std::size_t batch, const std::size_t row, const std::size_t col) {
    
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    if(r < row && c < col) {
        float b = *Get(B, r, c, pitchB);
        for(int i = 0;i < batch;i++) {
            *Get(C, r + i * row, c, pitchC) = *Get(A, r + i * row, c, pitchA) + b;
        }
    }
}
__global__ void PlusInplaceBatchKernel(
    float* A, const float* B,
    const std::size_t pitchA, const std::size_t pitchB,
    const std::size_t batch, const std::size_t row, const std::size_t col) {
    
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    if(r < row && c < col) {
        float bValue = *Get(B, r, c, pitchB);
        for(int i = 0;i < batch;i++) {
            *Get(A, r + i * row, c, pitchA) += bValue;
        }
    }
}
__global__ void PlusReduceInplaceBatchKernel(
    float* A, const float* B,
    const std::size_t pitchA, const std::size_t pitchB,
    const std::size_t batch, const std::size_t row, const std::size_t col) {
    
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    if(r < row && c < col) {
        float aValue = *Get(A, r, c, pitchA);
        for(int i = 0;i < batch;i++) {
            aValue += *Get(B, r + i * row , c, pitchB);
        }
        *Get(A, r, c, pitchA) = aValue;
    }
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
__global__ void MulInplaceKernel(
    float* A, const float x,
    const std::size_t pitchA,
    const std::size_t row, const std::size_t col) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    if(r < row && c < col) {
        *Get(A, r, c, pitchA) *= x;
    }
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



__global__ void SetKernel(
    float* A, const float x,
    const std::size_t pitchA,
    const std::size_t row, const std::size_t col) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    if(r < row && c < col) {
        *Get(A, r, c, pitchA) = x;
    }
}



__global__ void ReduceMaxKernel(
    const float* A, const float* C,
    const std::size_t pitchA, const std::size_t pitchC,
    const std::size_t row, const std::size_t col) {
}



__global__ void ApplyLookAheadMaskBatchKernel(
    float* A, const int seq, const float x,
    const std::size_t pitchA, 
    const std::size_t batch, const std::size_t row, const std::size_t col) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int sr = r % batch;

    if((c > sr || sr >= seq) && (r < row && c < col)) {
        *Get(A, r, c, pitchA) = x;
    }
}
__global__ void ApplyPaddingMaskBatchKernel(
    float* A, const int seq, const float x,
    const std::size_t pitchA, 
    const std::size_t batch, const std::size_t row, const std::size_t col) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int sr = r % batch;

    if((c >= seq || sr >= seq) && (r < row && c < col)) {
        *Get(A, r, c, pitchA) = x;
    }
}
__global__ void ApplyCrossPaddingMaskBatchKernel(
    float* A, const int seq, const float x,
    const std::size_t pitchA, 
    const std::size_t batch, const std::size_t row, const std::size_t col) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    if(c >= seq && (r < row && c < col)) {
        *Get(A, r, c, pitchA) = x;
    }
}



// Need -use_fast_math
__global__ void GetPositionalEncodeKernel(
    float* A, const std::size_t pitchA, 
    const std::size_t row, const std::size_t col) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    if(r < row && c < col) {
        if(c % 2 == 0) {
            *Get(A, r, c, pitchA) = __sinf(r * __frcp_rn(powf(10000.0f, float(c) / col)));
        }
        else {
            *Get(A, r, c, pitchA) = __cosf(r * __frcp_rn(powf(10000.0f, float(c - 1) / col)));
        }
    }
}



// A : d1 * d2
// B : d2 * d3
// C : d1 * d3
__global__ void MatMulKernelAB(
    const float* A, const float* B, float* C, const std::size_t pitchA, const std::size_t pitchB, const std::size_t pitchC,
    const std::size_t d1, const std::size_t d2, const std::size_t d3) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    float CValue = 0.0f;

    __shared__ float As[MATMUL_BLOCKSIZE][MATMUL_BLOCKSIZE];
    __shared__ float Bs[MATMUL_BLOCKSIZE][MATMUL_BLOCKSIZE];

    for(int i = 0;i < ceil(d2, MATMUL_BLOCKSIZE);i++) {
           
        const std::size_t loadIdxX = i * MATMUL_BLOCKSIZE + threadIdx.x;
        const std::size_t loadIdxY = i * MATMUL_BLOCKSIZE + threadIdx.y;

        if(r < d1 && loadIdxX < d2) {
            As[threadIdx.y][threadIdx.x] = Get(A,r,loadIdxX,pitchA)[0];
        }
        else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if(loadIdxY < d2 && c < d3) {
            Bs[threadIdx.y][threadIdx.x] = Get(B,loadIdxY,c,pitchB)[0];
        }
        else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        if(r < d1 && c < d3) {
            for(int j = 0;j < MATMUL_BLOCKSIZE;j++) {
                CValue += As[threadIdx.y][j] * Bs[j][threadIdx.x];
            }
        }
        __syncthreads();
    }
    if(r < d1 && c < d3) {
        Get(C,r,c,pitchC)[0] += CValue;
    }
}
// A : d2 * d1
// B : d2 * d3
// C : d1 * d3
__global__ void MatMulKernelATB(
    const float* A, const float* B, float* C, const std::size_t pitchA, const std::size_t pitchB, const std::size_t pitchC,
    const std::size_t d1, const std::size_t d2, const std::size_t d3) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    float CValue = 0.0f;

    __shared__ float As[MATMUL_BLOCKSIZE][MATMUL_BLOCKSIZE];
    __shared__ float Bs[MATMUL_BLOCKSIZE][MATMUL_BLOCKSIZE];

    for(int i = 0;i < ceil(d2, MATMUL_BLOCKSIZE);i++) {

        const std::size_t a0 = blockIdx.y * blockDim.y;
        const std::size_t b0 = blockIdx.x * blockDim.x;
        const std::size_t loadIdxY = i * MATMUL_BLOCKSIZE + threadIdx.y;

        if(loadIdxY < d2 && a0 + threadIdx.x < d1) {
            As[threadIdx.y][threadIdx.x] = *Get(A,loadIdxY,a0 + threadIdx.x,pitchA);
        }
        else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if(loadIdxY < d2 && b0 + threadIdx.x < d3) {
            Bs[threadIdx.y][threadIdx.x] = *Get(B,loadIdxY,b0 + threadIdx.x,pitchB);
        }
        else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        if(r < d1 && c < d3) { 
            for(int j = 0;j < MATMUL_BLOCKSIZE;j++) {
                CValue += As[j][threadIdx.y] * Bs[j][threadIdx.x];
            }
        }
        __syncthreads();
    }
    if(r < d1 && c < d3) {
        *Get(C,r,c,pitchC) += CValue;
    }
}
// A : d1 * d2
// B : d3 * d2
// C : d1 * d3
__global__ void MatMulKernelABT(
    const float* A, const float* B, float* C, const std::size_t pitchA, const std::size_t pitchB, const std::size_t pitchC,
    const std::size_t d1, const std::size_t d2, const std::size_t d3) {

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    float CValue = 0.0f;

    __shared__ float As[MATMUL_BLOCKSIZE][MATMUL_BLOCKSIZE];
    __shared__ float Bs[MATMUL_BLOCKSIZE][MATMUL_BLOCKSIZE];

    for(int i = 0;i < ceil(d2, MATMUL_BLOCKSIZE);i++) {

        const std::size_t a0 = blockIdx.y * blockDim.y;
        const std::size_t b0 = blockIdx.x * blockDim.x;
        const std::size_t loadIdxX = i * MATMUL_BLOCKSIZE + threadIdx.x;

        if(a0 + threadIdx.y < d1 && loadIdxX < d2) {
            As[threadIdx.y][threadIdx.x] = *Get(A,a0 + threadIdx.y,loadIdxX,pitchA);
        }
        else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if(b0 + threadIdx.y < d3 && loadIdxX < d2) {
            Bs[threadIdx.y][threadIdx.x] = *Get(B,b0 + threadIdx.y,loadIdxX,pitchB);
        }
        else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        if(r < d1 && c < d3) {
            for(int j = 0;j < MATMUL_BLOCKSIZE;j++) {
                CValue += As[threadIdx.y][j] * Bs[threadIdx.x][j];
            }
        }
        __syncthreads();
    }
    if(r < d1 && c < d3) {
        *Get(C,r,c,pitchC) += CValue;
    }
}