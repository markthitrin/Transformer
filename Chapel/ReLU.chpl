use Config;
use Util;
use Tensor;
use Timer;

class ReLU {
    
    proc init() {
        domMask = {0..#(batch * sequenceLength * dFF)};
    }

    proc forward(ref input: [] real(32), ref output: [] real(32)) : void {
        for i in 0..#(batch * sequenceLength * dFF) {
            output[i] = if input[i] >= 0 then input[i] else 0.0:real(32);
        }
        CheckPoint();
    }   

    proc predict(ref input: [] real(32), ref output: [] real(32)) : void {
        forward(input, output);
    }

    proc backward(ref outputGradient: [] real(32), ref inputGradient: [] real(32), ref input: [] real(32)) : void {
        for i in 0..#(batch * sequenceLength * dFF) {
            outputGradient[i] = if input[i] >= 0 then outputGradient[i] else 0.0:real(32);
        }
        Copy(0,0,batch * sequenceLength * dFF,outputGradient,inputGradient);
        CheckPoint();
    }

    var domMask: domain(1);
    var mask: [domMask] real(32);
}