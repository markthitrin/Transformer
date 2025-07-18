use Util;
use Math;
use Matrix;
use Config;
use CTypes;
use Timer;

class Embedding {

    proc init(in numTokens: int) {
        domTable = {0..#(numTokens * dModel)};
        table = 0;
        
        domTableOpt = {0..#(numTokens)};
        tableOpt = new AdamOptimizer({0..#dModel});

        UniformInit(table, 0.1);
    }

    proc forward(ref input: [?Di] int, ref output: [?Do] real(32)) : void {
        for i in 0..#(batch * sequenceLength) {
            Copy(input[i] * dModel, i * dModel, dModel, table, output);
        }
        Mul(0, 0, batch * sequenceLength * dModel, output, sqrt(dModel):real(32), output);
    }

    proc predict(ref input: [?Di] int, ref output: [?Do] real(32)) : void {
        forward(input, output);
    }

    proc backward(ref outputGradient: [?Do] real(32), ref input: [?Di] int) : void {
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

    proc loadParam() {
        loadM(table);
    }

    proc forwardTest() {
        var input: [0..#(batch * sequenceLength)] int;
        var output: [0..#(batch * sequenceLength * dModel)] real(32);
        var target: [0..#(batch * sequenceLength * dModel)] real(32);

        loadM(input);
        loadM(target);

        forward(input, output);

        PrintTestResult("forward", output, target);
    }

    proc checkUpdateParam() {
        var tableUpdated: [domTable] real(32);

        loadM(tableUpdated);

        PrintTestResult("backward table", table, tableUpdated);
    }

    proc backwardTest() {
        var input: [0..#(batch * sequenceLength)] int;
        var output: [0..#(batch * sequenceLength * dModel)] real(32);
        var target: [0..#(batch * sequenceLength * dModel)] real(32);
        var outputGradient: [0..(batch * sequenceLength * dModel)] real(32);

        outputGradient = (1.0 / outputGradient.domain.size):real(32);

        loadM(input);

        forward(input, output);
        backward(outputGradient, input);
        updateParameter();

        checkUpdateParam();
    }

    var domTable: domain(1);
    var table: [domTable] real(32);
    
    var domTableOpt: domain(1);
    var needUpdate: [domTableOpt] bool;
    var tableOpt: [domTableOpt] AdamOptimizer;
}


// Test code

// var model = new Embedding(srcVocab);
// model.loadParam();
// for i in 0..4 do model.forwardTest();
// for i in 0..4 do model.backwardTest();