use Util;
use Math;
use Config;
use DropOut;
use Tensor;
use Timer;

class PositionalEncoder {

    proc init() {
        dropout = new DropOut(batch * sequenceLength * dModel);

        domMask = {0..#(sequenceLength * dModel)};
        mask = 0;
        for i in 0..#sequenceLength {
            for j in 0..<dModel by 2 {
                mask[i * dModel + j] = sin(i:real(32) / 10000.0 ** ((j):real(32) / dModel)):real(32);
            }
            for j in 1..<dModel by 2 {
                mask[i * dModel + j] = cos(i:real(32) / 10000.0 ** ((j - 1):real(32) / dModel)):real(32);
            }
        }
    }

    proc forward(ref input: [] real(32), ref output: [] real(32)) : void {
        var block = sequenceLength * dModel;
        for i in 0..#batch {
            Plus(i * block, 0, i * block, block, input, mask, input);
        }
        CheckPoint();
        dropout.forward(input, output);
    }

    proc predict(ref input: [] real(32), ref output: [] real(32)) : void {
        var block = sequenceLength * dModel;
        for i in 0..#batch {
            Plus(i * block, 0, i * block, block, input, mask, input);
        }
        dropout.predict(input, output);
    }

    proc backward(ref outputGradient: [] real(32), ref inputGradient: [] real(32)) {
        dropout.backward(outputGradient, inputGradient);
        CheckPoint();
    }

    var dropout: owned DropOut;

    var domMask: domain(1);
    var mask: [domMask] real(32);
}