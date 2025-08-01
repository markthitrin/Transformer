use Config;
use Math;
use Random;
use Tensor;
use Timer;
use Util;

var rng = new randomStream(int, seed=0);

proc GenerateDropoutMask(in size: int, ref mask: [] int, in dropoutRate : real(32)) {
    var thresholdInt = (dropoutRate * 65535):int;
    rng.fill(mask, 0, 65535);
    for i in 0..#size {
        mask[i] = if mask[i] > thresholdInt then 1 else 0;
    }
}

class DropOut {
    
    proc init(in size: int) {
        domMask = {0..#(size)};
    }

    proc forward(ref input: [] real(32), ref output: [] real(32)) : void {
        GenerateDropoutMask(domMask.size, mask, dropoutRate);
        var corrector: real(32) = 1.0 / (1.0 - dropoutRate);
        for i in 0..#domMask.size {
            output[i] = input[i] * mask[i]:real(32) * corrector;
        }
        CheckPoint();
    }

    proc predict(ref input: [] real(32), ref output: [] real(32)) : void {
        output = input;
    }

    proc backward(ref outputGradient: [] real(32), ref inputGradient: [] real(32)) : void {
        var corrector: real(32) = (1.0 - dropoutRate);
        for i in 0..#domMask.size {
            inputGradient[i] = outputGradient[i] * mask[i]:real(32) * corrector;
        }
        CheckPoint();
    }

    var domMask: domain(1);
    var mask: [domMask] int;
}