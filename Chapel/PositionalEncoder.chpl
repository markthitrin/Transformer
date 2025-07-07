use Util;
use Math;
use Config;
use Matrix;
use DropOut;

class PositionalEncoder {
    proc init() {
        dropout = new DropOut(batch * sequenceLength, dModel);

        domMask = {0..#(sequenceLength * dModel)};
        for i in 0..#sequenceLength {
            for j in 0..<dModel by 2 {
                mask[i * dModel + j] = sin(i / 10000.0 ** (j / dModel));
            }
            for j in 1..<dModel by 2 {
                mask[i * dModel + j] = cos(i / 10000.0 ** (j / dModel));
            }
        }
    }

    proc forward(ref input: [?D] real, ref output: [D] real) : void {
        param block = sequenceLength * dModel;
        for i in 0..#batch {
            Mul(input[(i * block)..#block], mask, out1[(i * block)..#block]);
        }
        dropout.forward(out1, output);
    }

    proc predict(ref input: [?D] real, ref output: [D] real) : void {
        param block = sequenceLength * dModel;
        for i in 0..#batch {
            Mul(input[(i * block)..#block], mask, out1[(i * block)..#block]);
        }
        dropout.predict(out1, output);
    }

    var dropout: owned DropOut;

    var domMask: domain(1);
    var mask: [domMask] real(32);

    var out1: [domMask] real(32);
}