use Util;
use Config;
use Matrix;
use Timer;

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

    proc forward(ref input: [?Di] real(32), ref output: [?Do] real(32)) {
        var outD = bias.domain.size;
        var inD = weight.domain.size / bias.domain.size;
        var batch = Do.size / outD;
        forall i in BalancePar(0,batch, (batch * outD * 0.0005):real(32), 1, 2) {
            Copy(0, i * outD, outD, bias, output);
        }
        MatMulPlusABPar(batch, inD, outD, input, weight, output);
        CheckPoint();
    }

    proc predict(ref input: [?Di] real(32), ref output: [?Do] real(32)) {
        forward(input, output);
    }

    proc backward(
        ref outputGradient: [?Do] real(32), ref inputGradient: [?Di] real(32),
        ref input:[Di] real(32)) {

        var outD = bias.domain.size;
        var inD = weight.domain.size / bias.domain.size;
        var batch = Do.size / outD;
        inputGradient = 0;
        var temp: [0..#outD] real(32);
        forall i in BalancePar(0, batch, (batch * outD):real(32), 1, 2)
            with (+ reduce temp) {
            Plus(0, i * outD, 0, outD, temp, outputGradient, temp);
        }
        Plus(0, 0, 0, outD, biasOpt.gradient, temp, biasOpt.gradient);
        MatMulPlusATBPar(inD, batch, outD, input, outputGradient, weightOpt.gradient);
        MatMulPlusABT(batch, outD, inD, outputGradient, weight, inputGradient);
        CheckPoint();
    }

    proc updateParameterTask() {
        cobegin {
            AdamOpt(weight, weightOpt);
            AdamOpt(bias, biasOpt);
        }
    }

    proc checkUpdateParam() {
        var weightUpdated: [weight.domain] real(32);
        var biasUpdated: [bias.domain] real(32);

        loadM(weightUpdated);
        loadM(biasUpdated);

        PrintTestResult("backward weight", weight, weightUpdated);
        PrintTestResult("backward bias", bias, biasUpdated);
    }

    var domWeight: domain(1);
    var domBias: domain(1);
    var weight: [domWeight] real(32);
    var bias: [domBias] real(32);

    var weightOpt: AdamOptimizer;
    var biasOpt: AdamOptimizer;
}