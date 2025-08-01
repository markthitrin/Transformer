use Util;
use Math;
use Tensor;
use Timer;

class Softmax {

    proc init(in batch: int, in shape: int) {
        this.batch = batch;
        this.shape = shape;

        domB = {0..#shape};
    }

    proc forward(ref input: [] real(32), ref output: [] real(32)) : void {
        for i in 0..#batch {
            var maxValue: real(32);
            var sumExp: real(32);
            MaxReduce(i*shape, shape, input, maxValue);
            Exp(i * shape, 0, shape, input, maxValue, buffer);
            PlusReduce(0, shape, buffer, sumExp);
            Div(0, i * shape, shape, buffer, sumExp, output);
        }
        CheckPoint();
    }

    proc predict(ref input: [] real(32), ref output: [] real(32)) : void {
        forward(input, output);
    }

    proc backward(ref outputGradient: [] real(32), ref inputGradient: [] real(32),
        ref output: [] real(32)) : void {

        for i in 0..#batch {
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