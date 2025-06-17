#ifndef DROP_OUT
#define DROP_OUT

#include "Header.h"
#include "Tensor.h"

template<float dropoutRate, int row, int col>
void GenerateDropoutMask(Tensor<row, col> _mask) {
    IMPORT(mask);

    static thread_local uint32_t state = static_cast<uint32_t>(
        std::chrono::steady_clock::now().time_since_epoch().count() +
        reinterpret_cast<uintptr_t>(&state));

    auto fast_rand = [&]() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    };
    const uint32_t threshold = static_cast<uint32_t>((1.0f - dropoutRate) * 0xFFFFFFFFu);

    for (int i = 0; i < row * col; ++i) {
        mask[i] = (fast_rand() < threshold) ? 1.0f : 0.0f;
    }
}

template<int row,int col, float dropoutRate>
class DropOut {
public:
    DropOut() {
        _mask.init(); 
    }

    void forward() noexcept {
        constexpr float corrector = (1.0f - dropoutRate);
        GenerateDropoutMask<dropoutRate>(_mask);
        Mul(_input, _mask, _output);
        Div(_output, corrector, _output);
    }

    void predict() noexcept {
        Copy(_input, _output);
    }

    void backward() noexcept {
        Mul(_inGradient, _mask, _outGradient);
    }

    Tensor<row, col> _input;
    Tensor<row, col> _output;
    Tensor<row, col> _inGradient;
    Tensor<row, col> _outGradient;

    Tensor<row, col> _mask;
};

#endif // !DROP_OUT
