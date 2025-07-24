use Config;
use Util;
use Math;
use Random;
use Matrix;
use Timer;

var domRng = 1..0;
var rng: [domRng] randomStream(int);

proc GenerateDropoutMaskPar(in size: int, ref mask: [] int, in dropoutRate : real(32)) {
    var thresholdInt = (dropoutRate * 65535):int;
    // for i in D {
    //     mask[i] = rng.next(0, 65535);
    // }
    var numT = getNumThreads(size, (size * 0.0001):real(32), 10, 1);
    var chunkSize = (mask.domain.size + numT - 1) / numT;
    coforall i in 0..#numT {
        var start = chunkSize * i;
        var end = min(mask.domain.size, start + chunkSize);
        rng[i].fill(mask[start..<end], 0, 65535);
        for i in start..<end {
            mask[i] = if mask[i] > thresholdInt then 1 else 0;
        }
    }
}

class DropOut {
    proc init(in size: int) {
        domRng = 0..#numPar;
        domMask = {0..#(size)};
    }

    proc forward(ref input: [?D] real(32), ref output: [D] real(32)) : void {
        // GenerateDropoutMaskPar(domMask.size, mask, dropoutRate);
        // var corrector: real(32) = 1.0 / (1.0 - dropoutRate);
        // MulPar(0, 0, 0, domMask.size, input, mask, output);
        // DivPar(0, 0, domMask.size, output, corrector, output);
        // CheckPoint();

        DivPar(0, 0, domMask.size, input, 1.0 - dropoutRate, output);
    }

    proc predict(ref input: [?D] real(32), ref output: [D] real(32)) : void {
        output = input;
    }

    proc backward(ref outputGradient: [?D] real(32), ref inputGradient: [D] real(32)) : void {
        // var corrector: real(32) = (1.0 - dropoutRate);
        // MulPar(0, 0, 0, domMask.size, outputGradient, mask, inputGradient);
        // DivPar(0, 0, domMask.size, inputGradient, corrector, inputGradient);
        // CheckPoint();

        DivPar(0, 0, domMask.size, outputGradient, 1.0 - dropoutRate, inputGradient);
    }

    var domMask: domain(1);
    var mask: [domMask] int;
}