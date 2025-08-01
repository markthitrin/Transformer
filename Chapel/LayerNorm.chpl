use Config;
use Tensor;
use Timer;
use Util;

class LayerNorm {

    proc init() {
        domGB = {0..#(dModel)};
        alpha = 1.0;
        bias = 0.0;

        alphaOpt = new AdamOptimizer(alpha);
        biasOpt = new AdamOptimizer(bias);

        domXHat = {0..#(batch * sequenceLength * dModel)};
        domStd = {0..#(batch * sequenceLength * dModel)};
        xHat = 0;
        std = 0;
    }

    proc forward(ref input: [] real(32), ref output: [] real(32)) : void {

        forall i in Par(0, batch * sequenceLength, 52) {
            var mean: real(32);
            PlusReduce(i * dModel, dModel, input, mean);
            mean /= dModel;

            StdReduce((i * dModel), dModel, input, mean, std[i]);

            for j in 0..#dModel {
                xHat[i * dModel + j] = (input[i * dModel + j] - mean) / (std[i] + eps);
                output[i * dModel + j] = alpha[j] * xHat[i * dModel + j] + bias[j];
            }
        }
        CheckPoint();
    }

    proc predict(ref input: [] real(32), ref output: [] real(32)) : void {
        forward(input, output);
    }

    proc backward(ref outputGradient: [] real(32), ref inputGradient: [] real(32)) : void {
        var invD: real(32) = (1.0 / dModel):real(32);
        ref alphaGrad = alphaOpt.gradient;
        ref biasGrad = biasOpt.gradient;

        forall i in Par(0, batch * sequenceLength, 52)
            with (+ reduce alphaGrad, + reduce biasGrad) {
            var invStd: real(32) = (1.0 / std[i]): real(32);
            var sumG: real(32) = 0.0;
            var sumGXHat: real(32) = 0.0;
            for j in 0..#dModel {
                var gxH = outputGradient[i * dModel + j] * xHat[i * dModel + j];
                alphaGrad[j] += gxH;
                sumGXHat += gxH;
            }
            for j in 0..#dModel {
                biasGrad[j] += outputGradient[i * dModel + j];
                sumG += outputGradient[i * dModel + j];
            }
            var a: real(32) = sumG * invD;
            var b: real(32) = sumGXHat * invD;
            
            for j in 0..#dModel {
                inputGradient[i * dModel + j] = invStd * (outputGradient[i * dModel + j] - a - xHat[i * dModel + j] * b) * alpha[j];
            }
        }
        CheckPoint();
    }

    proc updateParameterTask() {
        cobegin {
            AdamOpt(alpha, alphaOpt);
            AdamOpt(bias, biasOpt);
        }
    }
    
    var domGB: domain(1);
    var alpha: [domGB] real(32);
    var bias: [domGB] real(32);

    var alphaOpt: AdamOptimizer;
    var biasOpt: AdamOptimizer;

    var domXHat: domain(1);
    var domStd: domain(1);
    var xHat: [domXHat] real(32);
    var std: [domStd] real(32);
}