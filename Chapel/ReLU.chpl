use Config;
use Util;
use Matrix;

class ReLU {
    proc init() {
        domMask = {0..#(batch * sequenceLength * dFF)};
    }

    proc forward(ref input: [?D] real(32), ref output: [D] real(32)) : void {
        for i in D {
            mask[i] = if input[i] >= 0 then 1.0:real(32) else 0.0:real(32);
        }
        Mul(input, mask, output);
    }   

    proc predict(ref input: [?D] real(32), ref output: [D] real(32)) : void {
        forward(input, output);
    }

    proc backward(ref outputGradient: [?D] real(32), ref inputGradient: [D] real(32)) : void {
        Mul(outputGradient, mask, inputGradient);
    }

    var domMask: domain(1);
    var mask: [domMask] real(32);
}