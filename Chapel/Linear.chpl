use Util;
use Config;
use Matrix;

class Linear {
    proc init(in inD: int, in outD: int) {
        domWeight = {0..#(inD * outD)};
        domBias = {0..#(outD)};
        HeNormalInit(weight);
        HeNormalInit(bias);
        
        weightOpt = new AdamOptGradient(weight);
        biasOpt = new AdamOptGradient(bias);

        this.inD = inD;
        this.outD = outD;
    }

    proc forward(ref input: [?Di] real(32), ref output: [?Do] real(32)) {
        var batch = Di.size / inD;
        MatMulAB(input, weight, output);
        for i in batch {
            Plus(output[(i * outD)..#outD], bias, output[(i * outD)..#outD]);
        }
    }

    proc predict(ref input: [?Di] real(32), ref output: [?Do] real(32)) {
        forward(input, output);
    }

    proc backward(ref outputGradient: [?Do] real(32), ref inputGradient: [?Di] real(32),
        ref input:[Di] real(32)) {

        var batch = Di.size / inD;
        for i in batch {
            Plus(biasOpt.gradient, outputGradient[(i * outD)..#outD], biasOpt.gradient);
        }
        MatMulPlusATB(input, outputGradient, weightOpt.gradient);
        MatMulPlusABT(outputGradient, weight, inputGradient);
    }

    proc updateParameter() {
        AdamOpt(weight, weightOpt);
        AdamOpt(bias, biasOpt);
    }

    var domWeight: domain(1);
    var domBias: domain(1);
    var weight: [domWeight] real(32);
    var bias: [domBias] real(32);

    var weightOpt: AdamOptGradient;
    var biasOpt: AdamOptGradient;

    var inD: int;
    var outD: int;
}