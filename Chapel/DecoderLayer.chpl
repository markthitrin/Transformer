use Config;
use DropOut;
use FeedForwardBlock;
use LayerNorm;
use Math;
use MultiheadAttention;
use Tensor;
use Timer;
use Util;

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
        ref input: [] real(32), ref encoderOut:[] real(32), ref output: [] real(32),
        ref srcSeq: [] int, ref tgtSeq: [] int) : void {
            
        norm1.forward(input, out1);
        mulAtt1.forward(out1, out1, out1, out2, MaskType.LOOK_AHEAD, tgtSeq);
        dropout1.forward(out2, out3);
        Plus(0, 0, 0, batch * sequenceLength * dModel, input, out3, out3);

        norm2.forward(out3, out4);
        mulAtt2.forward(out4, encoderOut, encoderOut, out5, MaskType.CROSS_PADDING, srcSeq);
        dropout2.forward(out5, out6);
        Plus(0, 0, 0, batch * sequenceLength * dModel, out3, out6, out6);

        norm3.forward(out6, out7);
        pff.forward(out7, out8);
        dropout3.forward(out8, output);
        Plus(0, 0, 0, batch * sequenceLength * dModel, out6, output, output);
    }

    proc predict(
        ref input: [] real(32), ref encoderOut:[] real(32), ref output: [] real(32),
        ref srcSeq: [] int, ref tgtSeq: [] int) : void {
            
        norm1.predict(input, out1);
        mulAtt1.predict(out1, out1, out1, out2, MaskType.LOOK_AHEAD, tgtSeq);
        dropout1.predict(out2, out3);
        Plus(0, 0, 0, batch * sequenceLength * dModel, input, out3, out3);

        norm2.predict(out3, out4);
        mulAtt2.predict(out4, encoderOut, encoderOut, out5, MaskType.CROSS_PADDING, srcSeq);
        dropout2.predict(out5, out6);
        Plus(0, 0, 0, batch * sequenceLength * dModel, out3, out6, out6);

        norm3.predict(out6, out7);
        pff.predict(out7, out8);
        dropout3.predict(out8, output);
        Plus(0, 0, 0, batch * sequenceLength * dModel, out6, output, output);
    }

    proc backward(
        ref outputGradient: [] real(32), ref encoderGradient: [] real(32), ref inputGradient: [] real(32),
        ref encoderOut: [] real(32), ref srcSeq: [] int, ref tgtSeq: [] int) : void {

        dropout3.backward(outputGradient, gradient8);
        pff.backward(gradient8, gradient7, out7);
        norm3.backward(gradient7, gradient6);
        Plus(0, 0, 0, batch * sequenceLength * dModel, outputGradient, gradient6, gradient6);

        dropout2.backward(gradient6, gradient5);
        mulAtt2.backward(gradient5, gradient4, encoderGradient, encoderGradient, out4, encoderOut, encoderOut, out5, MaskType.CROSS_PADDING, srcSeq);
        norm2.backward(gradient4, gradient3);
        Plus(0, 0, 0, batch * sequenceLength * dModel, gradient6, gradient3, gradient3);

        dropout1.backward(gradient3, gradient2);
        mulAtt1.backward(gradient2, gradient1, gradient1, gradient1, out1, out1, out1, out2, MaskType.LOOK_AHEAD, tgtSeq);
        norm1.backward(gradient1, inputGradient);
        Plus(0, 0, 0, batch * sequenceLength * dModel, gradient3, inputGradient, inputGradient);
    }

    proc updateParameter() {
        norm1.updateParameter();
        mulAtt1.updateParameter();
        norm2.updateParameter();
        mulAtt2.updateParameter();
        norm3.updateParameter();
        pff.updateParameter();
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