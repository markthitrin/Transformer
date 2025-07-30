use Config;
use Tensor;
use Random;
use Timer;

var rng = new randomStream(int);

proc GenerateDropoutMaskPar(in size: int, ref mask: [] int, in dropoutRate : real(32)) {
    var thresholdInt = (dropoutRate * 65535):int;
    rng.fill(mask, 0, 65535);
    forall i in Par(0, size, 24) {
        mask[i] = if mask[i] > thresholdInt then 1 else 0;
    }
}

class DropOut {
    proc init(in size: int) {
        domMask = {0..#(size)};
    }

    proc forward(ref input: [] real(32), ref output: [] real(32)) : void {
        // GenerateDropoutMaskPar(domMask.size, mask, dropoutRate);
        // var corrector: real(32) = 1.0 / (1.0 - dropoutRate);
        // forall i in Par(0, domMask.size, 24) {
        //     output[i] = mask[i]:real(32) * input[i];
        // }
        // DivPar(0, 0, domMask.size, output, corrector, output);
        // CheckPoint();

        DivPar(0, 0, domMask.size, input, 1.0 - dropoutRate, output);
    }

    proc predict(ref input: [] real(32), ref output: [] real(32)) : void {
        output = input;
    }

    proc backward(ref outputGradient: [] real(32), ref inputGradient: [] real(32)) : void {
        // var corrector: real(32) = (1.0 - dropoutRate);
        // forall i in Par(0, domMask.size, 24) {
        //     inputGradient[i] = mask[i]:real(32) * outputGradient[i];
        // }   
        // DivPar(0, 0, domMask.size, inputGradient, corrector, inputGradient);
        // CheckPoint();

        DivPar(0, 0, domMask.size, outputGradient, 1.0 - dropoutRate, inputGradient);
    }

    var domMask: domain(1);
    var mask: [domMask] int;
}