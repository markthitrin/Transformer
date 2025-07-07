use Util;
use Math;
use Random;
use Matrix;

var rng = new randomStream(real, seed=0);

proc GenerateDropoutMask(ref mask: [?D] real(32), in dropoutRate : real(32)) {
    fillRandom(mask, 0.0, 1.0);
    for i in D {
        mask[i] = if mask[i] > dropoutRate then 1.0 else 0.0;
    }
}

class DropOut {
    proc init(in size: int, in dropoutRate: real) {
        this.dropoutRate = dropoutRate;
        this.size = size;
    }

    proc forward(ref input: [?D] real(32), ref output: [D] real(32)) : [D] real {
        GenerateDropoutMask(mask, dropoutRate);
        Mul(input, mask, output);
        Div(output, 1.0 - dropoutRate, output);
    }

    proc predict(ref input: [?D] real(32), ref output: [D] real(32)) : [D] real {
        output = input;
    }

    proc backward(ref outputGradient: [?D] real(32), ref inputGradient: [D] real(32)) : void {
        Mul(outputGradient, mask, inputGradient);
        Div(inputGradient, 1.0 - dropoutRate, inputGradient);
    }    

    var dropoutRate: real(32);

    var domMask: domain(1);
    var mask: [domMask] real(32);

    var size: int;
}