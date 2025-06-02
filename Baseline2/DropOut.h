#ifndef DROP_OUT
#define DROP_OUT

#include "Header.h"
#include "Tensor.h"

template<float dropoutRate>
void GenerateDropoutMask(float* mask, int size) {
    static thread_local std::mt19937 rng{
        static_cast<unsigned>(
            std::chrono::steady_clock::now().time_since_epoch().count() +
            reinterpret_cast<uintptr_t>(&rng))
    };

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < size; ++i) {
        mask[i] = (dist(rng) < (1.0f - dropoutRate)) ? 1.0f : 0.0f;
    }
}

template<int d,int col, float dropoutRate>
class DropOut {
public:
    DropOut() noexcept : 
        _mask(Create<d,col>()) { 
    }
    ~DropOut() {
        _aligned_free(_mask);
    }

    void forward() noexcept {
        IMPORT_CONST(input);
        IMPORT(output);
        IMPORT(mask);
        constexpr float corrector = (1.0f - dropoutRate);
        GenerateDropoutMask<dropoutRate>(mask, d * col);
        Mul<d, col>((Tensor)input, (Tensor)mask, (Tensor)output);
        Div<d, col>((Tensor)output, corrector, (Tensor)output);
    }

    void predict() noexcept {
        IMPORT_CONST(input);
        IMPORT(output);
        Copy<d, col>((Tensor)input, (Tensor)output);
    }

    void backpropagate() noexcept {
        IMPORT_CONST(inGradient);
        IMPORT_CONST(mask);
        IMPORT(outGradient);
        Mul<d, col>((Tensor)inGradient, (Tensor)mask, (Tensor)outGradient);
    }

    Tensor _input;
    Tensor _output;
    Tensor _inGradient;
    Tensor _outGradient;

    Tensor _mask;
};

#endif // !DROP_OUT
