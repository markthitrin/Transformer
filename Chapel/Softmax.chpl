use Config;
use Util;
use Math;
use Tensor;
use Timer;

class Softmax {
    proc init() {
        domB = {0..#(batch * head * sequenceLength * sequenceLength)};
    }

    proc forward(ref input: [] real(32), ref output: [] real(32)) : void {
        forall i in 0..#(batch * head * sequenceLength) {
            var maxValue: real(32);
            var sumExp: real(32);

            MaxReduce(i * sequenceLength, sequenceLength, input, maxValue);
            Exp(i * sequenceLength, i * sequenceLength, sequenceLength, input, maxValue, buffer);
            PlusReduce(i * sequenceLength, sequenceLength, buffer, sumExp);
            Div(i * sequenceLength, i * sequenceLength, sequenceLength, buffer, sumExp, output);
        }
        CheckPoint();
    }

    proc predict(ref input: [] real(32), ref output: [] real(32)) : void {
        forward(input, output);
    }

    proc backward(ref outputGradient: [] real(32), ref inputGradient: [] real(32), ref output: [] real(32)) : void {
        forall i in 0..#(batch * head * sequenceLength) {
            var sumGY: real(32);
            ProductPlusReduce(i * sequenceLength, i * sequenceLength, sequenceLength, outputGradient, output, sumGY);
            
            for j in 0..#sequenceLength {
                inputGradient[i * sequenceLength + j] = output[i * sequenceLength + j] * (outputGradient[i * sequenceLength + j] - sumGY);
            }
        }
        CheckPoint();
    }

    var domB: domain(1);
    var buffer: [domB] real(32);
}