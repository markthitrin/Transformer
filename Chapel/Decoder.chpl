use Config;
use DecoderLayer;
use LayerNorm;
use Tensor;
use Timer;
use Util;

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
        ref input: [] real(32), ref encoderOut: [] real(32), ref output: [] real(32),
        ref srcSeq: [] int, ref tgtSeq: [] int) : void {

        layers[0]!.forward(input, encoderOut, outi[0], srcSeq, tgtSeq);
        for i in 1..<N {
            layers[i]!.forward(outi[i - 1], encoderOut, outi[i], srcSeq, tgtSeq);
        }
        norm.forward(outi[N - 1], output);
    }

    proc predict(
        ref input: [] real(32), ref encoderOut:[] real(32), ref output: [] real(32),
        ref srcSeq: [] int, ref tgtSeq: [] int) : void {

        layers[0]!.predict(input, encoderOut, outi[0], srcSeq, tgtSeq);
        for i in 1..<N {
            layers[i]!.predict(outi[i - 1], encoderOut, outi[i], srcSeq, tgtSeq);
        }
        norm.predict(outi[N - 1], output);
    }

    proc backward(
        ref outputGradient: [] real(32), ref inputGradient: [] real(32), ref encoderGradient: [] real(32), ref encoderOut: [] real(32),
        ref srcSeq: [] int, ref tgtSeq: [] int) : void {
        
        Set(0, encoderGradient.domain.size, encoderGradient, 0.0); // need to be set manually
        norm.backward(outputGradient, gradienti[N - 1]);
        for i in 1..(N - 1) by -1 {
            layers[i]!.backward(gradienti[i], encoderGradient, gradienti[i - 1], encoderOut, srcSeq, tgtSeq);
        }
        layers[0]!.backward(gradienti[0], encoderGradient, inputGradient, encoderOut, srcSeq, tgtSeq);
    }

    proc updateParameterTask() : void {
        cobegin {
            coforall i in 0..#N {
                layers[i]!.updateParameterTask();
            }
            norm.updateParameterTask();
        }
    }
    
    var domLayers: domain(1);
    var layers: [domLayers] owned DecoderLayer?;
    var norm: owned LayerNorm;

    var domOG: domain(1);
    var outi: [domLayers][domOG] real(32);
    var gradienti: [domLayers][domOG] real(32);
}
