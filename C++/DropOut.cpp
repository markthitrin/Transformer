#include "Config.h"
#include "DropOut.h"
#include "Header.h"
#include "Tensor.h"
#include "Timer.h"

struct pcg_setseq_64_xsh_rr_32 {
    uint64_t state;
    uint64_t inc;

    pcg_setseq_64_xsh_rr_32(uint64_t seed = 0x853c49e6748fea9bULL,
                             uint64_t seq  = 0xda3e39cb94b95bdbULL)
        : state(0), inc((seq << 1u) | 1u) {
        seed_rng(seed);
    }

    void seed_rng(uint64_t seed) {
        state = 0;
        next();
        state += seed;
        next();
    }

    uint32_t next() {
        uint64_t oldstate = state;
        state = oldstate * 6364136223846793005ULL + inc;
        uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
        uint32_t rot = static_cast<uint32_t>(oldstate >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }
};

std::vector<pcg_setseq_64_xsh_rr_32> rng;

void GenerateDropoutMask(TensorView mask, pcg_setseq_64_xsh_rr_32 rng) {
    int threshold = (1.0f - dropoutRate) * 65535;

    for (int i = 0; i < mask.row * mask.col; ++i) {
        mask[i] = ((rng.next() & 0xFFFF) < threshold) ? 1.0f : 0.0f;
    }
}

void GenerateDropoutMaskPar(TensorView mask) {
    const int numT = std::min(numPar, 64);
    #pragma omp parallel num_threads(numT)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        int chunk_size = (mask.row + nthreads - 1) / nthreads;
        int start = tid * chunk_size;
        int end = std::min(start + chunk_size, mask.row);
        if(start < mask.row) {
            GenerateDropoutMask(mask.sliceRow(start, end - start), tid);
        }
    }
}

DropOut::DropOut(const int row, const int col) : mask(row, col) {
    rng.emplace_back(std::rand());
}

void DropOut::forward(TensorView input, TensorView output) {
    GenerateDropoutMaskPar(mask);
    MulPar(input, mask, output);
    DivPar(output, (1.0f - dropoutRate), output);
    Timer::CheckPoint();
}

void DropOut::predict(TensorView input, TensorView output) {
    output = input;
}

void DropOut::backward(TensorView outputGradient, TensorView inputGradient) {
    MulPar(outputGradient, mask, inputGradient);
    DivPar(inputGradient, (1.0f - dropoutRate), inputGradient);
    Timer::CheckPoint();
}