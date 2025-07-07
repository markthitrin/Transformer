use Util;
use Math;
use Matrix;
use Config;

class Embedding {
    proc init(in numTokens: int) {
        domTable = {0..#(numTokens * ..#dModel)};
        UniformInit(table, 0.1);

        domTableOpt = {0..#numTokens};
        tableOpt = new AdamOptGradient(domTable);
    }

    proc forward(ref input: [?Di] real(32), ref output: [?Do] real(32)) : void {
        for i in 0..#(batch * sequenceLength) {
            output[(i * dModel)..#dMoe] = table[(input[i] * dModel)..#dModel];
        }
        Mul(output, sqrt(dModel), output);
    }

    proc predict(ref input: [?Di] real(32), ref output: [?Do] real(32)) : void {
        return forward(input, output);
    }

    proc backward(ref input: [?Di] real(32), ref outputGradient: [?Do] real(32)) : void {
        Mul(outputGradient, sqrt(dModel), outputGradient);
        for i in 0..#(batch * sequenceLength) {
            Plus(tableOpt.gradient[(input[i] * dModel)..#dModel], outputGradient[(i * dModel)..#dModel], tableOpt.gradient[(input[i] * dModel)..#dModel]);
        }
    }

    proc updateParameter(ref input: [?Di] real(32)) {
        for i in 0..#(batch * sequenceLength) {
            if tableOpt.gradient[input[i] * dModel] != 0 then
                AdamOpt(table, tableOpt);
        }
    }

    var domTable: domain(1);
    var table: [domTable] real(32);
    
    var domTableOpt: domain(1);
    var tableOpt: AdamOptGradient;
}