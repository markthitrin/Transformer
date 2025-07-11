use Util;
use Matrix;
use Config;

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

    proc forward(ref input: [?D] real(32), ref output: [D] real(32)) : void {
        for i in 0..#(batch * sequenceLength) {
            var mean: real(32);
            PlusReduce(input[(i * dModel)..#dModel], mean);
            mean /= dModel;

            StdReduce(input[(i * dModel)..#dModel], mean, std[i]);

            for j in 0..#dModel {
                xHat[i * dModel + j] = (input[i * dModel + j] - mean) / (std[i] + eps);
                output[i * dModel + j] = alpha[j] * xHat[i * dModel + j] + bias[j];
            }
        }
    }

    proc predict(ref input: [?D] real(32), ref output: [D] real(32)) : void {
        forward(input, output);
    }

    proc backward(ref outputGradient: [?D] real(32), ref inputGradient: [D] real(32)) : void {
        for i in 0..#(batch * sequenceLength) {
            PlusProductInplace(alphaOpt.gradient, outputGradient[(i * dModel)..#dModel], xHat[(i * dModel)..#dModel]);
            Plus(biasOpt.gradient, outputGradient[(i * dModel)..#dModel], biasOpt.gradient);
            var sumG: real(32) = 0.0;
            var sumGXHat: real(32) = 0.0;
            PlusReduce(outputGradient[(i * dModel)..#dModel], sumG);
            ProductPlusReduce(outputGradient[(i * dModel)..#dModel], xHat[(i * dModel)..#dModel], sumGXHat);

            var a: real(32) = sumG / dModel;
            var b: real(32) = sumGXHat / dModel;
            
            for j in 0..#dModel {
                inputGradient[i * dModel + j] = (1.0 / std[i]) * (outputGradient[i * dModel + j] - a - xHat[i * dModel + j] * b) * alpha[j];
            }
        }
    }

    proc updateParameter() {
        AdamOpt(alpha, alphaOpt);
        AdamOpt(bias, biasOpt);
    }

    proc loadParam() {
        loadM(alpha);
        loadM(bias);
    }

    proc forwardTest() {
        var input: [0..#(batch * sequenceLength * dModel)] real(32);
        var output: [0..#(batch * sequenceLength * dModel)] real(32);
        var target: [0..#(batch * sequenceLength * dModel)] real(32);

        loadM(input);
        loadM(target);

        forward(input, output);

        PrintTestResult("forward", output, target);
    }

    proc checkUpdateParam() {
        var alphaUpdated: [domGB] real(32);
        var biasUpdated: [domGB] real(32);

        loadM(alphaUpdated);
        loadM(biasUpdated);

        PrintTestResult("backward alpha", alpha, alphaUpdated);
        PrintTestResult("backward bias", bias, biasUpdated);
    }

    proc backwardTest() {
        var input: [0..#(batch * sequenceLength * dModel)] real(32);
        var output: [0..#(batch * sequenceLength * dModel)] real(32);
        var target: [0..#(batch * sequenceLength * dModel)] real(32);
        var outputGradient: [0..(batch * sequenceLength * dModel)] real(32);
        var inputGradient: [0..(batch * sequenceLength * dModel)] real(32);

        outputGradient = (1.0 / outputGradient.domain.size):real(32);

        loadM(input);

        forward(input, output);
        backward(outputGradient, inputGradient);
        updateParameter();

        checkUpdateParam();
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

// Test code
// var model = new LayerNorm();
// model.loadParam();
// for i in 0..4 do model.forwardTest();
// for i in 0..4 do model.backwardTest();