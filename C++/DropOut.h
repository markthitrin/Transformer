#ifndef DROP_OUT
#define DROP_OUT

#include "Header.h"
#include "Tensor.h"

void GenerateDropoutMask(TensorView mask);

class DropOut {
public:
    DropOut(const int row, const int col);

    void forward(TensorView input, TensorView output);

    void predict(TensorView input, TensorView output);

    void backward(TensorView outputGradient, TensorView inputGradient);

    Tensor mask;
};

#endif
