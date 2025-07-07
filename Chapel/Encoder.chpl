use Util;
use Math;
use Config;
use Matrix;
use EncoderLayer;
use LayerNorm;


class Encoder {
    proc init() {
        domLayer = {0..#N};
        for i in 0..#N {
            layers[i] = new EncoderLayer();
        }
        norm = new LayerNorm();

        domOut = {0..#(batch * sequenceLength * dModel)};
        domGradient = {0..#(batch * sequenceLength * dModel)};
    }

    proc forward(ref input: [?D] real(32), ref output: [D] real(32)) : void {
        layers[0].forward(input, outT[0]);
        for i in 1..<N {
            layers[i].forward(outT[i - 1],outT[i]);
        }
        norm.forward(outT[N - 1, output]);
    }

    proc predict(ref input: [?D] real(32), ref output: [D] real(32)) : void {
        layers[0].predict(input, outT[0]);
        for i in 1..<N {
            layers[i].predict(outT[i - 1],outT[i]);
        }
        norm.predict(outT[N - 1, output]);
    }

    proc backward(ref outputGradient: [?D] real(32), ref inputGradient: [D] real(32)) : void {
        norm.backward(outputGradient, gradientT[N - 1]);
        for i in (N - 1)..1 by -1 {
            layers[i].backward(gradientT[i], gradientT[i - 1]);
        }
        layers[0].backward(gradientT[0], inputGradient);
    }

    proc updateParameter() {
        for i in 0..#N {
            layers[i].updateParameter();
        }
        norm.updateParameter();
     }
    
    var domLayer: domain(1);
    var layers: [domLayers] owned EncoderLayer;
    var norm: owned LayerNorm;

    var domOut: domain(1);
    var domGradient: domain(1);
    var outT: [domLayer][domOut] real(32);
    var gradientT: [domLayer][domOut] real(32);
}