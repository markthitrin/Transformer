use Util;
use Math;
use Config;
use Matrix;
use DropOut;

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

    proc forward(ref input: [?D] real(32), ref output: [D] real(32)) : void {
        param block = sequenceLength * dModel;
        for i in 0..#batch {
            Plus(input[(i * block)..#block], mask, input[(i * block)..#block]);
        }
        dropout.forward(input, output);
    }

    proc predict(ref input: [?D] real(32), ref output: [D] real(32)) : void {
        param block = sequenceLength * dModel;
        for i in 0..#batch {
            Plus(input[(i * block)..#block], mask, input[(i * block)..#block]);
        }
        dropout.predict(input, output);
    }

    proc backward(ref outputGradient: [?D] real(32), ref inputGradient: [D] real(32)) {
        dropout.backward(outputGradient, inputGradient);
    }

    proc forwardTest() {
        var input: [0..#(batch * sequenceLength * dModel)] real(32);
        var output: [0..#(batch * sequenceLength * dModel)] real(32);
        var target: [0..#(batch * sequenceLength * dModel)] real(32);

        loadM(input);
        loadM(target);

        forward(input, output);

        PrintTestResult("forward", output, target);
    }

    var dropout: owned DropOut;

    var domMask: domain(1);
    var mask: [domMask] real(32);
}

// Test code
// var model = new PositionalEncoder();
// for i in 0..4 do model.forwardTest();