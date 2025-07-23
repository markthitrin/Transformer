use Config;
use Util;
use Math;
use Random;
use Matrix;
use Timer;

var rng = new randomStream(int, seed=0);

proc GenerateDropoutMask(in size: int, ref mask: [] int, in dropoutRate : real(32)) {
    var thresholdInt = (dropoutRate * 65535):int;
    // for i in D {
    //     mask[i] = rng.next(0, 65535);
    // }
    rng.fill(mask, 0, 65535);
    forall i in 0..#size {
        mask[i] = if mask[i] > thresholdInt then 1 else 0;
    }
}

class DropOut {
    proc init(in size: int) {
        domMask = {0..#(size)};
    }

    proc forward(ref input: [?D] real(32), ref output: [D] real(32)) : void {
        GenerateDropoutMask(domMask.size, mask, dropoutRate);
        var corrector: real(32) = 1.0 / (1.0 - dropoutRate);
        forall i in D {
            output[i] = input[i] * mask[i]:real(32) * corrector;
        }
        CheckPoint();

        // Div(0, 0, domMask.size, input, 1.0 - dropoutRate, output);
    }

    proc predict(ref input: [?D] real(32), ref output: [D] real(32)) : void {
        output = input;
    }

    proc backward(ref outputGradient: [?D] real(32), ref inputGradient: [D] real(32)) : void {
        var corrector: real(32) = (1.0 - dropoutRate);
        forall i in D {
            inputGradient[i] = outputGradient[i] * mask[i]:real(32) * corrector;
        }
        CheckPoint();

        // Div(0, 0, domMask.size, outputGradient, 1.0 - dropoutRate, inputGradient);
    }

    var domMask: domain(1);
    var mask: [domMask] int;
}