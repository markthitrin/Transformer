use Util;
use Math;
use Config;
use Matrix;
use DecoderLayer;
use LayerNorm;
use Timer;

class Decoder {

    proc init() {
        domLayers = {0..#N};
        layers = [i in 0..#N] nil;
        for i in domLayers {
            layers[i] = new DecoderLayer();
        }
        norm = new LayerNorm();

        domOG = {0..#(batch * sequenceLength * dModel)};
    }

    proc forward(
        ref input: [?D] real(32), ref encoderOut:[D] real(32), ref output: [D] real(32),
        ref srcSeq: [?Ds] int, ref tgtSeq: [Ds] int) : void {
        layers[0]!.forward(input, encoderOut, outi[0], srcSeq, tgtSeq);
        for i in 1..<N {
            layers[i]!.forward(outi[i - 1], encoderOut, outi[i], srcSeq, tgtSeq);
        }
        norm.forward(outi[N - 1], output);
    }

    proc predict(
        ref input: [?D] real(32), ref encoderOut:[D] real(32), ref output: [D] real(32),
        ref srcSeq: [?Ds] int, ref tgtSeq: [Ds] int) : void {
        layers[0]!.predict(input, encoderOut, outi[0], srcSeq, tgtSeq);
        for i in 1..<N {
            layers[i]!.predict(outi[i - 1], encoderOut, outi[i], srcSeq, tgtSeq);
        }
        norm.predict(outi[N - 1], output);
    }

    proc backward(
        ref outputGradient: [?D] real(32), ref inputGradient: [D] real(32), ref encoderGradient: [D] real(32), ref encoderOut: [D] real(32),
        ref srcSeq: [?Ds] int, ref tgtSeq: [Ds] int) : void {
        Set(encoderGradient, 0.0); // need manually reset;
        norm.backward(outputGradient, gradienti[N - 1]);
        for i in 1..(N - 1) by -1 {
            layers[i]!.backward(gradienti[i], encoderGradient, gradienti[i - 1], encoderOut, srcSeq, tgtSeq);
        }
        layers[0]!.backward(gradienti[0], encoderGradient, inputGradient, encoderOut, srcSeq, tgtSeq);
    }

    proc updateParameter() {
        for i in 0..#N {
            layers[i]!.updateParameter();
        }
        norm.updateParameter();
    }

    proc loadParam() {
        for i in 0..#N {
            layers[i]!.loadParam();
        }
        norm.loadParam();
    }

    proc forwardTest() {
        var input1: [0..#(batch * sequenceLength * dModel)] real(32);
        var input2: [0..#(batch * sequenceLength * dModel)] real(32);
        var output: [0..#(batch * sequenceLength * dModel)] real(32);
        var target: [0..#(batch * sequenceLength * dModel)] real(32);
        var npdLoader: [0..#2] real(32);
        var srcSeq: [0..#batch] int;
        var tgtSeq: [0..#batch] int;

        loadM(input1);
        loadM(input2);
        loadM(target);
        loadM(npdLoader);
        for i in 0..#batch do srcSeq[i] = npdLoader[0]:int;
        for i in 0..#batch do tgtSeq[i] = npdLoader[1]:int;

        forward(input1, input2, output, srcSeq, tgtSeq);

        PrintTestResult("forward", output, target);
    }

    proc checkUpdateParam() {
        for i in 0..#N {
            layers[i]!.checkUpdateParam();
        }
        norm.checkUpdateParam();
    }

    proc backwardTest() {
        var input1: [0..#(batch * sequenceLength * dModel)] real(32);
        var input2: [0..#(batch * sequenceLength * dModel)] real(32);
        var output: [0..#(batch * sequenceLength * dModel)] real(32);
        var target: [0..#(batch * sequenceLength * dModel)] real(32);
        var outputGradient: [0..#(batch * sequenceLength * dModel)] real(32);
        var inputGradient: [0..#(batch * sequenceLength * dModel)] real(32);
        var npdLoader: [0..#2] real(32);
        var srcSeq: [0..#batch] int;
        var tgtSeq: [0..#batch] int;
        
        outputGradient = (1.0 / outputGradient.domain.size):real(32);

        loadM(input1);
        loadM(input2);
        loadM(target);
        loadM(npdLoader);
        for i in 0..#batch do srcSeq[i] = npdLoader[0]:int;
        for i in 0..#batch do tgtSeq[i] = npdLoader[1]:int;

        forward(input1, input2, output, srcSeq, tgtSeq);
        backward(outputGradient, inputGradient, inputGradient, input2, srcSeq, tgtSeq);
        updateParameter();

        checkUpdateParam();
    }
    
    var domLayers: domain(1);
    var layers: [domLayers] owned DecoderLayer?;
    var norm: owned LayerNorm;

    var domOG: domain(1);
    var outi: [domLayers][domOG] real(32);
    var gradienti: [domLayers][domOG] real(32);
}

// Test code
// var model = new Decoder();
// model.loadParam();
// for i in 0..4 do model.forwardTest();
// for i in 0..4 do model.backwardTest();