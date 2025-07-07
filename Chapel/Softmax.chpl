use Util;
use Math;
use Matrix;


class Softmax {
    proc init(in batch: int, in shape: int) {
        this.batch = batch;
        this.shape = shape;
    }

    proc forward(ref input: [?D] real(32), ref output: [D] real(32)) : void {
        for i in 0..#batch {
            var maxValue: real(32);
            var sumExp: real(32);
            MaxReduce(input[(i * shape)..#shape], maxValue);
            ExpPlusReduce(input[(i * shape)..#shape], maxValue, sumExp);
            
            for j in 0..#shape {
                output[i * col + j] = exp(input[i * col + j] - maxValue) / sumExp;
            }
        }
    }

    proc predict(ref input: [?D] real(32), ref output: [D] real(32)) : void {
        return forward(input, output);
    }

    proc backward(ref outputGradient: [?D] real, ref inputGradient: [D] real,
        ref output: [D] real) : void {

        for i in 0..#batch {
            var sumGY: real(32);
            ProductPlusReduce(outputGradient[(i * shape)..#shape], output[(i * shape)..#shape], sumGY);
            
            for j in 0..#shape {
                inputGradient[i * shape + j] = output[i * shape + j] * (outputGradient[i * col + j] - sumGY);
            }
        }
    }

    var domOutput: domain(2);
    var output: [domOutput] real;

    var batch: int;
    var shape: int;
}