#include "Header.cuh"
#include "Tensor.cuh"
#include "TensorKernel.cuh"
#include "Util.cuh"
#include "UtilKernel.cuh"



void CopyAsync(Tensor& A, Tensor& B) {
    cudaMemcpy2DAsync(B.data, B.pitch, A.data, A.pitch, sizeof(float) * B.col, B.row, cudaMemcpyDeviceToDevice);
}
void CopyBatchAsync(Tensor& A, Tensor& B, const std::size_t batch) {
    const int sr = B.row / batch;
    for(int i = 0;i < batch;i++) {
        cudaMemcpy2DAsync(Get(B.data, i * sr, 0, B.pitch), B.pitch, A.data, A.pitch, sizeof(float) * A.col, A.row, cudaMemcpyDeviceToDevice);
    }
}



void Plus(Tensor& A, Tensor& B, Tensor& C) {
    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(A.col, BLOCKSIZE), ceil(A.row, BLOCKSIZE));
    PlusKernel<<<gridDim, blockDim>>>(A.data, B.data, C.data, A.pitch, B.pitch, C.pitch, A.row, A.col);
}
void Plus(Tensor& A, const float x, Tensor& C) {
    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(A.col, BLOCKSIZE), ceil(A.row, BLOCKSIZE));
    PlusKernel<<<gridDim, blockDim>>>(A.data, x, C.data, A.pitch, C.pitch, A.row, A.col);
}
void PlusBatch(Tensor& A, Tensor& B, Tensor& C, const std::size_t batch) { // C = A + b
    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(B.col, BLOCKSIZE), ceil(B.row, BLOCKSIZE));
    PlusBatchKernel<<<gridDim, blockDim>>>(A.data, B.data, C.data, A.pitch, B.pitch, C.pitch, batch, B.row, B.col);
}
void PlusInplaceBatchReduce(Tensor& A, Tensor& C, const std::size_t batch) { // C += a
    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(C.col, BLOCKSIZE), ceil(C.row, BLOCKSIZE));
    PlusInplaceBatchKernel<<<gridDim, blockDim>>>(A.data, C.data, A.pitch, C.pitch, batch, C.row, C.col);
}



void Sub(Tensor& A, Tensor& B, Tensor& C) {
    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(A.col, BLOCKSIZE), ceil(A.row, BLOCKSIZE));
    SubKernel<<<gridDim, blockDim>>>(A.data, B.data, C.data, A.pitch, B.pitch, C.pitch, A.row, A.col);
}
void Sub(Tensor& A, const float x, Tensor& C) {
    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(A.col, BLOCKSIZE), ceil(A.row, BLOCKSIZE));
    SubKernel<<<gridDim, blockDim>>>(A.data, x, C.data, A.pitch, C.pitch, A.row, A.col);
}



void Mul(Tensor& A, Tensor& B, Tensor& C) {
    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(A.col, BLOCKSIZE), ceil(A.row, BLOCKSIZE));
    MulKernel<<<gridDim, blockDim>>>(A.data, B.data, C.data, A.pitch, B.pitch, C.pitch, A.row, A.col);
}
void Mul(Tensor& A, const float x, Tensor& C) {
    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(A.col, BLOCKSIZE), ceil(A.row, BLOCKSIZE));
    MulKernel<<<gridDim, blockDim>>>(A.data, x, C.data, A.pitch, C.pitch, A.row, A.col);
}



void Div(Tensor& A, Tensor& B, Tensor& C) {
    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(A.col, BLOCKSIZE), ceil(A.row, BLOCKSIZE));
    DivKernel<<<gridDim, blockDim>>>(A.data, B.data, C.data, A.pitch, B.pitch, C.pitch, A.row, A.col);
}
void Div(Tensor& A, const float x, Tensor& C) {
    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(A.col, BLOCKSIZE), ceil(A.row, BLOCKSIZE));
    MulKernel<<<gridDim, blockDim>>>(A.data, 1.0f / x, C.data, A.pitch, C.pitch, A.row, A.col);
}



void Set(Tensor& A, const float x) {
    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(A.col, BLOCKSIZE), ceil(A.row, BLOCKSIZE));
    SetKernel<<<gridDim, blockDim>>>(A.data, x, A.pitch, A.row, A.col);
}
void Reset(Tensor& A) {
   cudaMemset2DAsync(A.data, A.pitch, 0, sizeof(float) * A.col, A.row);
}


void ReduceMax(Tensor& A, Tensor& C) {
    dim3 blockDim(REDUCTION_BLOCKSIZE_X, REDUCTION_BLOCKSIZE_Y);
    dim3 gridDim(ceil(1, REDUCTION_BLOCKSIZE_X), ceil(A.row, REDUCTION_BLOCKSIZE_Y));
    ReduceMaxKernel<<<gridDim,blockDim>>>( A.data, C.data, A.pitch, A.row, A.col );
}
void ReduceSumExp(Tensor& input, Tensor& maxValue, Tensor& sumExp) {
    dim3 blockDim(REDUCTION_BLOCKSIZE_X, REDUCTION_BLOCKSIZE_Y);
    dim3 gridDim(ceil(1, REDUCTION_BLOCKSIZE_X), ceil(input.row, REDUCTION_BLOCKSIZE_Y));
    ReduceSumExpKernel<<<gridDim, blockDim>>>(input.data, maxValue.data, sumExp.data,
        input.pitch,
        input.row, input.col);
}



void GetPositionalEncode(Tensor& A) {
    constexpr int BLOCKSIZE = 16;
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(ceil(A.col, BLOCKSIZE), ceil(A.row, BLOCKSIZE));
    GetPositionalEncodeKernel<<<gridDim, blockDim>>>(A.data, A.pitch, A.row, A.col);
}



void Print(Tensor& A) {
    float* _data = (float*)malloc(sizeof(float) * A.row * A.col);
    cudaMemcpy2D(_data, sizeof(float) * A.col, A.data, A.pitch, sizeof(float) * A.col, A.row, cudaMemcpyDeviceToHost);

    for (int i = 0; i < A.row; i++) {
        for (int j = 0; j < A.col; j++) {
            std::cout << _data[i * A.col + j] << " ";
        }
        std::cout << std::endl;
    }
}



void MatMulPlus(Tensor& A,Tensor& B, Tensor& C, bool ATransposed, bool BTransposed) {
    if(!ATransposed && !BTransposed) {
        dim3 blockDim(MATMUL_BLOCKSIZE, MATMUL_BLOCKSIZE);
        dim3 gridDim(ceil(C.col, MATMUL_BLOCKSIZE), ceil(C.row, MATMUL_BLOCKSIZE));
        MatMulKernelAB<<<gridDim, blockDim>>>(A.data, B.data, C.data, A.pitch, B.pitch, C.pitch, C.row, A.col, B.col);
    }
    else if(ATransposed && !BTransposed) {
        dim3 blockDim(MATMUL_BLOCKSIZE, MATMUL_BLOCKSIZE);
        dim3 gridDim(ceil(C.col, MATMUL_BLOCKSIZE), ceil(C.row, MATMUL_BLOCKSIZE));
        MatMulKernelATB<<<gridDim, blockDim>>>(A.data, B.data, C.data, A.pitch, B.pitch, C.pitch, C.row, A.row, C.col);
    }
    else if (!ATransposed && BTransposed) {
        dim3 blockDim(MATMUL_BLOCKSIZE, MATMUL_BLOCKSIZE);
        dim3 gridDim(ceil(C.col, MATMUL_BLOCKSIZE), ceil(C.row, MATMUL_BLOCKSIZE));
        MatMulKernelABT<<<gridDim, blockDim>>>(A.data, B.data, C.data, A.pitch, B.pitch, C.pitch, C.row, A.col, C.col);
    }
    else {
        // nothing implemented here.
    }
}