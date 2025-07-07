use Util;
use Matrix;
use Config;

class LayerNorm {

    proc init() {
        domGamma = {0..#(dModel)};
        domBias = {0..#(dModel)};
        gamma = 1.0;
        bias = 0.0;

        gammaOpt = new AdamOptGradient(gamma);
        biasOpt = new AdamOptGradient(bias);

        domXHat = {0..#(batch * sequenceLength * dModel)};
        domStd = {0..#(batch * sequenceLength * dModel)};
    }

    proc forward(ref input: [?D] real(32), ref output: [D] real(32)) : void {
        for i in 0..#batch {
            var mean: real(32);
            PlusReduce(A[(i * dModel)..#dModel], mean);
            mean /= dModel;

            StdReduce(A[(i * dModel)..#dModel], std[i]);

            for j in 0..#dModel {
                xHat[i * dModel + j] = (input[i * dModel + j] - mean) / (std[i] + eps);
                output[i * dModel + j] = alpha[j] * xHat[i * dModel + j] + bias[j];
            }
        }
    }

    proc predict(ref input: [?D] real(32), ref output: [D] real(32)) : void {
        return forward(tensor);
    }

    proc backward(ref outputGradient: [?D] real, ref inputGradient: [D]) : void {
        for i in 0..#batch {
            PlusProductInplace(alphaOpt.gradient, outputGradient[(i * dModel)..#dModel], xHat(i * dModel)..#dModel);
            Plus(biasOpt.gradient, outputGradient[(i * dModel)..#dModel], biasOpt.gradient);
            var sumG: real(32) = 0.0;
            var sumGXHat: real(32) = 0.0;
            PlusReduce(outputGradient[(i * dModel)..#dModel], sumG);
            PlusProductInplace(outputGradient[(i * dModel)..#dModel], xHat[(i * dModel)..#dModel], sumGXHat);

            var a: real(32) = sumG / dModel;
            var b: real(32) = sumGXHat / dModel;
            
            for j in 0..#dModel {
                inputGradient[i * dModel + j] = (1.0 / std[i]) * (outputGradient[i * col + j] - a - xHat[i * col + j] * b) * alpha[j];
            }
        }
    }

    proc updateParameter() {
        AdamOpt(gamma, gammaOpt);
        AdamOpt(bias, biasOpt);
    }
    
    var domGamma: domain(1);
    var domBias: domain(1);
    var gamma: [domGamma] real;
    var bias: [domBias] real;

    var gammaOpt: AdamOptGradient;
    var biasOpt: AdamOptGradient;

    var domXHat: domain(1);
    var domStd: domain(1);
    var xHat: [domXHat] real;
    var std: [domStd] real;
}
