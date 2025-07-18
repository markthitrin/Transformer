use Config;
use Util;
use Matrix;
use Timer;

class ReLU {
    proc init() {
        domMask = {0..#(batch * sequenceLength * dFF)};
    }

    proc forward(ref input: [?D] real(32), ref output: [D] real(32)) : void {
        for i in D {
            output[i] = if input[i] >= 0 then input[i] else 0.0:real(32);
        }
        CheckPoint();
    }   

    proc predict(ref input: [?D] real(32), ref output: [D] real(32)) : void {
        forward(input, output);
    }

    proc backward(ref outputGradient: [?D] real(32), ref inputGradient: [D] real(32), ref input: [D] real(32)) : void {
        for i in D {
            outputGradient[i] = if input[i] >= 0 then outputGradient[i] else 0.0:real(32);
        }
        Copy(0,0,D.size,outputGradient,inputGradient);
    }

    var domMask: domain(1);
    var mask: [domMask] real(32);
}