use Util;
use Math;
use Config;
use LayerNorm;
use MultiheadAttention;
use DropOut;
use FeedForwardBlock;
use Matrix;
use Timer;

class DecoderLayer {
    proc init() {
        norm1 = new LayerNorm();
        mulAtt1 = new MultiheadAttention();
        dropout1 = new DropOut(batch * sequenceLength * dModel);
        norm2 = new LayerNorm();
        mulAtt2 = new MultiheadAttention();
        dropout2 = new DropOut(batch * sequenceLength * dModel);
        norm3 = new LayerNorm();
        pff = new FeedForwardBlock();
        dropout3 = new DropOut(batch * sequenceLength * dModel);

        domOG = {0..#(batch * sequenceLength * dModel)};
    }

    proc forward(
        ref input: [?D] real(32), ref encoderOut:[D] real(32), ref output: [D] real(32),
        ref srcSeq: [?Ds] int, ref tgtSeq: [Ds] int) : void {
            
        norm1.forward(input, out1);
        mulAtt1.forward(out1, out1, out1, out2, MaskType.LOOK_AHEAD, tgtSeq);
        dropout1.forward(out2, out3);
        Plus(input, out3, out3);

        norm2.forward(out3, out4);
        mulAtt2.forward(out4, encoderOut, encoderOut, out5, MaskType.CROSS_PADDING, srcSeq);
        dropout2.forward(out5, out6);
        Plus(out3, out6, out6);

        norm3.forward(out6, out7);
        pff.forward(out7, out8);
        dropout3.forward(out8, output);
        Plus(out6, output, output);
    }

    proc predict(
        ref input: [?D] real(32), ref encoderOut:[D] real(32), ref output: [D] real(32),
        ref srcSeq: [?Ds] int, ref tgtSeq: [Ds] int) : void {
            
        norm1.predict(input, out1);
        mulAtt1.predict(out1, out1, out1, out2, MaskType.LOOK_AHEAD, tgtSeq);
        dropout1.predict(out2, out3);
        Plus(input, out3, out3);

        norm2.predict(out3, out4);
        mulAtt2.predict(out4, encoderOut, encoderOut, out5, MaskType.CROSS_PADDING, srcSeq);
        dropout2.predict(out5, out6);
        Plus(out3, out6, out6);

        norm3.predict(out6, out7);
        pff.predict(out7, out8);
        dropout3.predict(out8, output);
        Plus(out6, output, output);
    }

    proc backward(
        ref outputGradient: [?D] real(32), ref encoderGradient: [D] real(32), ref inputGradient: [D] real(32),
        ref encoderOut: [D] real(32), ref srcSeq: [?Ds] int, ref tgtSeq: [Ds] int) : void {

        dropout3.backward(outputGradient, gradient8);
        pff.backward(gradient8, gradient7, out7);
        norm3.backward(gradient7, gradient6);
        Plus(outputGradient, gradient6, gradient6);

        dropout2.backward(gradient6, gradient5);
        mulAtt2.backward(gradient5, gradient4, encoderGradient, encoderGradient, out4, encoderOut, encoderOut, out5, MaskType.CROSS_PADDING, srcSeq);
        norm2.backward(gradient4, gradient3);
        Plus(gradient6, gradient3, gradient3);

        dropout1.backward(gradient3, gradient2);
        mulAtt1.backward(gradient2, gradient1, gradient1, gradient1, out1, out1, out1, out2, MaskType.LOOK_AHEAD, tgtSeq);
        norm1.backward(gradient1, inputGradient);
        Plus(gradient3, inputGradient, inputGradient);
    }

    proc updateParameter() {
        norm1.updateParameter();
        mulAtt1.updateParameter();
        norm2.updateParameter();
        mulAtt2.updateParameter();
        norm3.updateParameter();
        pff.updateParameter();
    }

    proc loadParam() {
        norm1.loadParam();
        mulAtt1.loadParam();
        norm2.loadParam();
        mulAtt2.loadParam();
        norm3.loadParam();
        pff.loadParam();
    }

    proc forwardTest() {
        var input1: [0..#(batch * sequenceLength * dModel)] real(32);
        var input2: [0..#(batch * sequenceLength * dModel)] real(32);
        var output: [0..#(batch * sequenceLength * dModel)] real(32);
        var target: [0..#(batch * sequenceLength * dModel)] real(32);
        var npdLoader: [0..#2] real(32);
        var srcSeq: [0..#batch] int;
        var tgtSeq: [0..#batch] int;

        loadM(input1);
        loadM(input2);
        loadM(target);
        loadM(npdLoader);
        for i in 0..#batch do srcSeq[i] = npdLoader[0]:int;
        for i in 0..#batch do tgtSeq[i] = npdLoader[1]:int;

        forward(input1, input2, output, srcSeq, tgtSeq);

        PrintTestResult("forward", output, target);
    }

    proc checkUpdateParam() {
        norm1.checkUpdateParam();
        mulAtt1.checkUpdateParam();
        norm2.checkUpdateParam();
        mulAtt2.checkUpdateParam();
        norm3.checkUpdateParam();
        pff.checkUpdateParam();
    }

    proc backwardTest() {
        var input1: [0..#(batch * sequenceLength * dModel)] real(32);
        var input2: [0..#(batch * sequenceLength * dModel)] real(32);
        var output: [0..#(batch * sequenceLength * dModel)] real(32);
        var target: [0..#(batch * sequenceLength * dModel)] real(32);
        var outputGradient: [0..#(batch * sequenceLength * dModel)] real(32);
        var inputGradient: [0..#(batch * sequenceLength * dModel)] real(32);
        var npdLoader: [0..#2] real(32);
        var srcSeq: [0..#batch] int;
        var tgtSeq: [0..#batch] int;
        
        outputGradient = (1.0 / outputGradient.domain.size):real(32);

        loadM(input1);
        loadM(input2);
        loadM(npdLoader);
        for i in 0..#batch do srcSeq[i] = npdLoader[0]:int;
        for i in 0..#batch do tgtSeq[i] = npdLoader[1]:int;

        forward(input1, input2, output, srcSeq, tgtSeq);
        backward(outputGradient, inputGradient, inputGradient, input2, srcSeq, tgtSeq);
        updateParameter();

        checkUpdateParam();
    }

    var norm1: owned LayerNorm;
    var mulAtt1: owned MultiheadAttention;
    var dropout1: owned DropOut;
    var norm2: owned LayerNorm;
    var mulAtt2: owned MultiheadAttention;
    var dropout2: owned DropOut;
    var norm3: owned LayerNorm;
    var pff: owned FeedForwardBlock;
    var dropout3: owned DropOut;

    var domOG: domain(1);
    var out1: [domOG] real(32);
    var out2: [domOG] real(32);
    var out3: [domOG] real(32);
    var out4: [domOG] real(32);
    var out5: [domOG] real(32);
    var out6: [domOG] real(32);
    var out7: [domOG] real(32);
    var out8: [domOG] real(32);
    var gradient1: [domOG] real(32);
    var gradient2: [domOG] real(32);
    var gradient3: [domOG] real(32);
    var gradient4: [domOG] real(32);
    var gradient5: [domOG] real(32);
    var gradient6: [domOG] real(32);
    var gradient7: [domOG] real(32);
    var gradient8: [domOG] real(32);
}

// Test code
// var model = new DecoderLayer();
// model.loadParam();
// for i in 0..4 do model.forwardTest();
// for i in 0..4 do model.backwardTest();