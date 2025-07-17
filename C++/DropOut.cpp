#include "Header.h"
#include "Tensor.h"
#include "DropOut.h"
#include "Timer.h"

struct pcg_setseq_64_xsh_rr_32 {
    uint64_t state;
    uint64_t inc;  // Must be odd

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
} rng;

void GenerateDropoutMask(TensorView mask) {
    // static thread_local uint32_t state = static_cast<uint32_t>(
    //     std::chrono::steady_clock::now().time_since_epoch().count() +
    //     reinterpret_cast<uintptr_t>(&state));

    // auto fast_rand = [&]() {
    //     state ^= state << 13;
    //     state ^= state >> 17;
    //     state ^= state << 5;
    //     return state;
    // };
    // const uint32_t threshold = static_cast<uint32_t>((1.0f - dropoutRate) * 0xFFFFFFFFu);
    int threshold = (1.0f - dropoutRate) * 65535;

    for (int i = 0; i < mask.row * mask.col; ++i) {
        mask[i] = ((rng.next() & 0xFFFF) < threshold) ? 1.0f : 0.0f;
    }
}


DropOut::DropOut(const int row, const int col) : mask(row, col) {;}

void DropOut::forward(TensorView input, TensorView output) {
    // GenerateDropoutMask(mask);
    // Mul(input, mask, output);
    // Div(output, (1.0f - dropoutRate), output);

    Div(input, (1.0 - dropoutRate), output);
}

void DropOut::predict(TensorView input, TensorView output) {
    output = input;
}

void DropOut::backward(TensorView outputGradient, TensorView inputGradient) {
    // Mul(outputGradient, mask, inputGradient);
    // Div(inputGradient, (1.0f - dropoutRate), inputGradient);
    // Timer::CheckPoint();
    // if(verbose) std::cout << "DropOut" << std::endl;

    Div(outputGradient, (1.0 - dropoutRate), inputGradient);
}