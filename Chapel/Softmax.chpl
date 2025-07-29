use Util;
use Math;
use Matrix;
use Timer;

class Softmax {
    proc init(in batch: int, in shape: int) {
        this.batch = batch;
        this.shape = shape;

        domB = {0..#(shape * batch)};
    }

    proc forward(ref input: [?D] real(32), ref output: [D] real(32)) : void {
        forall i in 0..#batch {
            var maxValue: real(32);
            var sumExp: real(32);

            MaxReduce(i*shape, shape, input, maxValue);
            Exp(i * shape, i * shape, shape, input, maxValue, buffer);
            PlusReduce(i * shape, shape, buffer, sumExp);
            Div(i * shape, i * shape, shape, buffer, sumExp, output);
        }
        // CheckPoint();
    }

    proc predict(ref input: [?D] real(32), ref output: [D] real(32)) : void {
        forward(input, output);
    }

    proc backward(ref outputGradient: [?D] real(32), ref inputGradient: [D] real(32),
        ref output: [D] real(32)) : void {

        forall i in 0..#batch {
            var sumGY: real(32);
            ProductPlusReduce(i * shape, i * shape, shape, outputGradient, output, sumGY);
            
            for j in 0..#shape {
                inputGradient[i * shape + j] = output[i * shape + j] * (outputGradient[i * shape + j] - sumGY);
            }
        }
        CheckPoint();
    }

    var batch: int;
    var shape: int;

    var domB: domain(1);
    var buffer: [domB] real(32);
}