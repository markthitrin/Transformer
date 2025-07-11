use Util;
use Math;
use Config;
use Matrix;
use EncoderLayer;
use LayerNorm;


class Encoder {
    proc init() {
        domLayers = {0..#N};
        layers = [i in 0..#N] nil;
        for i in domLayers {
            layers[i] = new EncoderLayer();
        }
        norm = new LayerNorm();

        domOG = {0..#(batch * sequenceLength * dModel)};
    }

    proc forward(ref input: [?D] real(32), ref output: [D] real(32), ref srcSeq: [?Ds] int) : void {
        layers[0]!.forward(input, outi[0], srcSeq);
        for i in 1..<N {
            layers[i]!.forward(outi[i - 1], outi[i], srcSeq);
        }
        norm.forward(outi[N - 1], output);
    }

    proc predict(ref input: [?D] real(32), ref output: [D] real(32), ref srcSeq: [?Ds] int) : void {
        layers[0]!.predict(input, outi[0], srcSeq);
        for i in 1..<N {
            layers[i]!.predict(outi[i - 1], outi[i], srcSeq);
        }
        norm.predict(outi[N - 1], output);
    }

    proc backward(ref outputGradient: [?D] real(32), ref inputGradient: [D] real(32), ref srcSeq: [?Ds] int) : void {
        norm.backward(outputGradient, gradienti[N - 1]);
        for i in 1..(N - 1) by -1 {
            layers[i]!.backward(gradienti[i], gradienti[i - 1], srcSeq);
        }
        layers[0]!.backward(gradienti[0], inputGradient, srcSeq);
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
        var input: [0..#(batch * sequenceLength * dModel)] real(32);
        var output: [0..#(batch * sequenceLength * dModel)] real(32);
        var target: [0..#(batch * sequenceLength * dModel)] real(32);
        var npdLoader: [0..#1] real(32);
        var seq: [0..#batch] int;

        loadM(input);
        loadM(target);
        loadM(npdLoader);
        for i in 0..#batch do seq[i] = npdLoader[0]:int;

        forward(input, output, seq);

        PrintTestResult("forward", output, target);
    }

    proc checkUpdateParam() {
        for i in 0..#N {
            layers[i]!.checkUpdateParam();
        }
        norm.checkUpdateParam();
    }

    proc backwardTest() {
        var input: [0..#(batch * sequenceLength * dModel)] real(32);
        var output: [0..#(batch * sequenceLength * dModel)] real(32);
        var target: [0..#(batch * sequenceLength * dModel)] real(32);
        var outputGradient: [0..#(batch * sequenceLength * dModel)] real(32);
        var inputGradient: [0..#(batch * sequenceLength * dModel)] real(32);
        var npdLoader: [0..#1] real(32);
        var seq: [0..#batch] int;
        
        outputGradient = (1.0 / outputGradient.domain.size):real(32);

        loadM(input);
        loadM(npdLoader);
        for i in 0..#batch do seq[i] = npdLoader[0]:int;

        forward(input, output, seq);
        backward(outputGradient, inputGradient, seq);
        updateParameter();

        checkUpdateParam();
    }
    
    var domLayers: domain(1);
    var layers: [domLayers] owned EncoderLayer?;
    var norm: owned LayerNorm;

    var domOG: domain(1);
    var outi: [domLayers][domOG] real(32);
    var gradienti: [domLayers][domOG] real(32);
}

// Test code
// var model = new Encoder();
// model.loadParam();
// for i in 0..4 do model.forwardTest();
// for i in 0..4 do model.backwardTest();