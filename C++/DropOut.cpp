#include "Header.h"
#include "Tensor.h"
#include "DropOut.h"

void GenerateDropoutMask(TensorView mask) {
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

    for (int i = 0; i < mask.row * mask.col; ++i) {
        mask[i] = (fast_rand() < threshold) ? 1.0f : 0.0f;
    }
}


DropOut::DropOut(const int row, const int col) : mask(row, col) {;}

void DropOut::forward(TensorView input, TensorView output) {
    GenerateDropoutMask(mask);
    Mul(input, mask, output);
    Div(output, (1.0f - dropoutRate), output);

    // Div(input, (1.0 - dropoutRate), output);
}

void DropOut::predict(TensorView input, TensorView output) {
    output = input;
}

void DropOut::backward(TensorView outputGradient, TensorView inputGradient) {
    Mul(outputGradient, mask, inputGradient);
    Div(inputGradient, (1.0f - dropoutRate), inputGradient);

    // Div(outputGradient, (1.0 - dropoutRate), inputGradient);
}