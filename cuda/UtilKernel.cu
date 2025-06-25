#include "Header.cuh"
#include "UtilKernel.cuh"

__host__ __device__ int ceil(const int a, const int b) {
    return (a + b - 1) / b;
}

__global__ void AdamOptKernel(
    float* param, float* gradient, float* accM, float* accV, std::size_t* t,
    const std::size_t pitchParam, const std::size_t pitchGrad, const std::size_t pitchAccM, const std::size_t pitchAccV,
    const std::size_t row, const std::size_t col) {
   

    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    printf("%p thread %d %d ",(void*)t,r,c);

    const float invPowBeta1 = __frcp_rn(1.0f - powf(beta1, *t));
    const float invPowBeta2 = __frcp_rn(1.0f - powf(beta2, 1));
    if(r < row && c < col) {
        const float g = *Get(gradient, r, c, pitchGrad);
        *Get(gradient, r, c, pitchGrad) = 0;
        *Get(accM, r, c, pitchAccM) = *Get(accM, r, c, pitchAccM) * beta1 + g * (1.0f - beta1);
        *Get(accV, r, c, pitchAccV) = *Get(accV, r, c, pitchAccV) * beta2 + g * g * (1.0f - beta2);
        float mHat = *Get(accM, r, c, pitchAccM) * invPowBeta1;
        float vHat = *Get(accV, r, c, pitchAccV) * invPowBeta2;
        *Get(param, r, c, pitchParam) -= lr * mHat / (sqrtf(vHat) + eps);
    }
    __syncthreads();
    // if(r == 0 && c == 0) {
    //     (*t)++; 
    // }
}