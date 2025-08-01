use Config;
use DropOut;
use Linear;
use Math;
use Tensor;
use ReLU;
use Timer;
use Util;

class FeedForwardBlock {

    proc init() {
        linear1 = new Linear(dModel, dFF);
        relu = new ReLU();
        dropout = new DropOut(batch * sequenceLength * dFF);
        linear2 = new Linear(dFF, dModel);

        domOG = {0..#(batch * sequenceLength * dFF)};
    }

    proc forward(ref input: [] real(32), ref output: [] real(32)) : void {
        linear1.forward(input, out1);
        relu.forward(out1, out2);
        dropout.forward(out2, out3);
        linear2.forward(out3, output);
    }

    proc predict(ref input: [] real(32), ref output: [] real(32)) : void {
        linear1.predict(input, out1);
        relu.predict(out1, out2);
        dropout.predict(out2, out3);
        linear2.predict(out3, output);
    }

    proc backward(ref outputGradient: [] real(32), ref inputGradient: [] real(32), ref input: [] real(32)) : void {
        linear2.backward(outputGradient, gradient3, out3);
        dropout.backward(gradient3, gradient2);
        relu.backward(gradient2, gradient1, out1);
        linear1.backward(gradient1, inputGradient, input);
    }

    proc updateParameterTask() {
        cobegin {
            linear1.updateParameterTask();
            linear2.updateParameterTask();
        }
    }

    var linear1: owned Linear;
    var relu: owned ReLU;
    var dropout: owned DropOut;
    var linear2: owned Linear;

    var domOG: domain(1);
    var out1: [domOG] real(32);
    var out2: [domOG] real(32);
    var out3: [domOG] real(32);
    var gradient1: [domOG] real(32);
    var gradient2: [domOG] real(32);
    var gradient3: [domOG] real(32);
}