use Config;
use Tensor;
use Timer;
use Util;

class Linear {

    proc init(in inD: int, in outD: int) {
        domWeight = {0..#(inD * outD)};
        domBias = {0..#(outD)};
        weight = 0;
        bias = 0;
        HeNormalInit(weight);
        HeNormalInit(bias);
        
        weightOpt = new AdamOptimizer(weight);
        biasOpt = new AdamOptimizer(bias);
    }

    proc forward(ref input: [] real(32), ref output: [] real(32)) {
        var outD = bias.domain.size;
        var inD = weight.domain.size / bias.domain.size;
        var batch = output.domain.size / outD;
        for i in 0..#batch {
            Copy(0, i * outD, outD, bias, output);
        }
        MatMulPlusAB(batch, inD, outD, input, weight, output);
        CheckPoint();
    }

    proc predict(ref input: [] real(32), ref output: [] real(32)) {
        forward(input, output);
    }

    proc backward(
        ref outputGradient: [] real(32), ref inputGradient: [] real(32),
        ref input:[] real(32)) {

        var outD = bias.domain.size;
        var inD = weight.domain.size / bias.domain.size;
        var batch = outputGradient.domain.size / outD;
        inputGradient = 0;
        for i in 0..#batch {
            Plus(0, i * outD, 0, outD, biasOpt.gradient, outputGradient, biasOpt.gradient);
        }
        MatMulPlusATB(inD, batch, outD, input, outputGradient, weightOpt.gradient);
        MatMulPlusABT(batch, outD, inD, outputGradient, weight, inputGradient);
        CheckPoint();
    }

    proc updateParameter() {
        AdamOpt(weight, weightOpt);
        AdamOpt(bias, biasOpt);
    }

    var domWeight: domain(1);
    var domBias: domain(1);
    var weight: [domWeight] real(32);
    var bias: [domBias] real(32);

    var weightOpt: AdamOptimizer;
    var biasOpt: AdamOptimizer;
}