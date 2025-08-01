use Config;
use EncoderLayer;
use LayerNorm;
use Tensor;

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

    proc forward(ref input: [] real(32), ref output: [] real(32), ref srcSeq: [] int) : void {
        layers[0]!.forward(input, outi[0], srcSeq);
        for i in 1..<N {
            layers[i]!.forward(outi[i - 1], outi[i], srcSeq);
        }
        norm.forward(outi[N - 1], output);
    }

    proc predict(ref input: [] real(32), ref output: [] real(32), ref srcSeq: [] int) : void {
        layers[0]!.predict(input, outi[0], srcSeq);
        for i in 1..<N {
            layers[i]!.predict(outi[i - 1], outi[i], srcSeq);
        }
        norm.predict(outi[N - 1], output);
    }

    proc backward(ref outputGradient: [] real(32), ref inputGradient: [] real(32), ref srcSeq: [] int) : void {
        norm.backward(outputGradient, gradienti[N - 1]);
        for i in 1..(N - 1) by -1 {
            layers[i]!.backward(gradienti[i], gradienti[i - 1], srcSeq);
        }
        layers[0]!.backward(gradienti[0], inputGradient, srcSeq);
    }

    proc updateParameterTask() : void {
        for i in 0..#N {
            layers[i]!.updateParameterTask();
        }
        norm.updateParameterTask();
    }
    
    var domLayers: domain(1);
    var layers: [domLayers] owned EncoderLayer?;
    var norm: owned LayerNorm;

    var domOG: domain(1);
    var outi: [domLayers][domOG] real(32);
    var gradienti: [domLayers][domOG] real(32);
}