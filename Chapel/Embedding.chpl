use Config;
use CTypes;
use Math;
use Tensor;
use Timer;
use Util;

class Embedding {

    proc init(in numTokens: int) {
        domTable = {0..#(numTokens * dModel)};
        table = 0;
        
        domTableOpt = {0..#(numTokens)};
        tableOpt = new AdamOptimizer({0..#dModel});

        UniformInit(table, 0.1);
    }

    proc forward(ref input: [] int, ref output: [] real(32)) : void {
        for i in 0..#(batch * sequenceLength) {
            Copy(input[i] * dModel, i * dModel, dModel, table, output);
        }
        Mul(0, 0, batch * sequenceLength * dModel, output, sqrt(dModel):real(32), output);
        CheckPoint();
    }

    proc predict(ref input: [] int, ref output: [] real(32)) : void {
        forward(input, output);
    }

    proc backward(ref outputGradient: [] real(32), ref input: [] int) : void {
        Mul(0, 0, batch * sequenceLength * dModel, outputGradient, sqrt(dModel):real(32), outputGradient);
        for i in 0..#(batch * sequenceLength) {
            needUpdate[input[i]] = true;
            Plus(0, i*dModel, 0, dModel,
                tableOpt[input[i]].gradient,
                outputGradient,
                tableOpt[input[i]].gradient);
        }
        CheckPoint();
    }

    proc updateParameter() {
        for i in domTableOpt {
            if needUpdate[i] {
                AdamOpt(table[(i * dModel)..#dModel], tableOpt[i]);
                needUpdate[i] = false;
            }
        }
    }

    var domTable: domain(1);
    var table: [domTable] real(32);
    
    var domTableOpt: domain(1);
    var needUpdate: [domTableOpt] bool;
    var tableOpt: [domTableOpt] AdamOptimizer;
}