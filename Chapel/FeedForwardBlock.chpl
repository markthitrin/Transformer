use Util;
use Math;
use Linear;
use DropOut;
use ReLU;
use Config;
use Matrix;

class FeedForwardBlock {
    proc init() {
        linear1 = new Linear(dModel, dFF);
        relu = new ReLU();
        dropout = new DropOut(batch * sequenceLength * dFF, dropoutRate);
        linear2 = new Linear(dFF, dModel);

        domOut = {0..#(batch * sequenceLength * dModel)};
        domGradient = {0..#(batch * sequenceLength * dModel)};
    }

    proc forward(ref input: [?D] real(32), ref output: [D] real(32)) : void {
        linear1.forward(input, out1);
        relu.forward(out1, out2);
        dropout.forward(out2, out3);
        linear2.forward(out3, output);
    }

    proc predict(ref input: [?D] real(32), ref output: [D] real(32)) : void {
        linear1.predict(input, out1);
        relu.predict(out1, out2);
        dropout.predict(out2, out3);
        linear2.predict(out3, output);
    }

    proc backward(ref outputGradient: [?D] real(32), ref inputGradient: [D] real(32)) : void {
        linear2.backward(outputGradient, gradient3);
        dropout.backward(gradient3, gradient2);
        relu.backward(gradient2, gradient1);
        linear1.backward(gradient1, inputGradient);
    }

    proc updateParameter() {
        linear1.updateParameter();
        linear2.updateParameter();
    }

    var linear1: owned Linear;
    var relu: owned ReLU;
    var dropout: owned DropOut;
    var linear2: owned Linear;

    var domOut: domain(1);
    var domGradient: domain(1);
    var out1: [domOut] real(32);
    var out2: [domOut] real(32);
    var out3: [domOut] real(32);
    var gradient1: [domGradient] real(32);
    var gradient2: [domGradient] real(32);
    var gradient3: [domGradient] real(32);
}