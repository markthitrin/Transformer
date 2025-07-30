use Config;
use DropOut;
use FeedForwardBlock;
use LayerNorm;
use Tensor;
use MultiheadAttention;
use Timer;
use Util;

class EncoderLayer {
    proc init() {
        norm1 = new LayerNorm();
        mulAtt = new MultiheadAttention();
        dropout1 = new DropOut(batch * sequenceLength * dModel);
        norm2 = new LayerNorm();
        pff = new FeedForwardBlock();
        dropout2 = new DropOut(batch * sequenceLength * dModel);
        
        domOG = {0..#(batch * sequenceLength * dModel)};
    }

    proc forward(ref input: [] real(32), ref output: [] real(32), ref srcSeq: [] int) : void {
        norm1.forward(input, out1);
        mulAtt.forward(out1, out1, out1, out2, MaskType.PADDING, srcSeq);
        dropout1.forward(out2, out3);
        Plus(0, 0, 0, batch * sequenceLength * dModel, input, out3, out3);

        norm2.forward(out3, out4);
        pff.forward(out4, out5);
        dropout2.forward(out5, output);
        Plus(0, 0, 0, batch * sequenceLength * dModel, out3, output, output);
    }

    proc predict(ref input: [] real(32), ref output: [] real(32), ref srcSeq: [] int) : void {
        norm1.predict(input, out1);
        mulAtt.predict(out1, out1, out1, out2, MaskType.PADDING, srcSeq);
        dropout1.predict(out2, out3);
        Plus(0, 0, 0, batch * sequenceLength * dModel, input, out3, out3);

        norm2.predict(out3, out4);
        pff.predict(out4, out5);
        dropout2.predict(out5, output);
        Plus(0, 0, 0, batch * sequenceLength * dModel, out3, output, output);
    }

    proc backward(ref outputGradient: [] real(32), ref inputGradient: [] real(32), ref srcSeq: [] int) : void {
        dropout2.backward(outputGradient, gradient5);
        pff.backward(gradient5, gradient4, out4);
        norm2.backward(gradient4, gradient3);
        Plus(0, 0, 0, batch * sequenceLength * dModel, outputGradient, gradient3, gradient3);

        dropout1.backward(gradient3, gradient2);
        mulAtt.backward(gradient2, gradient1, gradient1, gradient1, out1, out1, out1, out2, MaskType.PADDING, srcSeq);
        norm1.backward(gradient1, inputGradient);
        Plus(0, 0, 0, batch * sequenceLength * dModel, gradient3, inputGradient, inputGradient);
    }

    proc updateParameterTask() : void {
        cobegin {
            norm1.updateParameterTask();
            mulAtt.updateParameterTask();
            norm2.updateParameterTask();
            pff.updateParameterTask();
        }
    }

    proc loadParam() {
        norm1.loadParam();
        mulAtt.loadParam();
        norm2.loadParam();
        pff.loadParam();
    }

    proc forwardTest() {
        var input: [0..#(batch * sequenceLength * dModel)] real(32);
        var output: [0..#(batch * sequenceLength * dModel)] real(32);
        var target: [0..#(batch * sequenceLength * dModel)] real(32);
        var npdLoader: [0..#1] real(32);
        var seq: [0..#batch] int;

        loadM(input);
        loadM(target);
        loadM(npdLoader);
        for i in 0..#batch do seq[i] = npdLoader[0]:int;

        forward(input, output, seq);

        PrintTestResult("forward", output, target);
    }

    proc checkUpdateParam() {
        norm1.checkUpdateParam();
        mulAtt.checkUpdateParam();
        norm2.checkUpdateParam();
        pff.checkUpdateParam();
    }

    proc backwardTest() {
        var input: [0..#(batch * sequenceLength * dModel)] real(32);
        var output: [0..#(batch * sequenceLength * dModel)] real(32);
        var target: [0..#(batch * sequenceLength * dModel)] real(32);
        var outputGradient: [0..#(batch * sequenceLength * dModel)] real(32);
        var inputGradient: [0..#(batch * sequenceLength * dModel)] real(32);
        var npdLoader: [0..#1] real(32);
        var seq: [0..#batch] int;
        
        outputGradient = (1.0 / outputGradient.domain.size):real(32);

        loadM(input);
        loadM(npdLoader);
        for i in 0..#batch do seq[i] = npdLoader[0]:int;

        forward(input, output, seq);
        backward(outputGradient, inputGradient, seq);
        updateParameterTask();

        checkUpdateParam();
    }

    var norm1: owned LayerNorm;
    var mulAtt: owned MultiheadAttention;
    var dropout1: owned DropOut;
    var norm2: owned LayerNorm;
    var pff: owned FeedForwardBlock;
    var dropout2: owned DropOut;

    var domOG: domain(1);
    var out1: [domOG] real(32);
    var out2: [domOG] real(32);
    var out3: [domOG] real(32);
    var out4: [domOG] real(32);
    var out5: [domOG] real(32);
    var gradient1: [domOG] real(32);
    var gradient2: [domOG] real(32);
    var gradient3: [domOG] real(32);
    var gradient4: [domOG] real(32);
    var gradient5: [domOG] real(32);
}

// Test code
// var model = new EncoderLayer();
// model.loadParam();
// for i in 0..4 do model.forwardTest();
// for i in 0..4 do model.backwardTest();