use Util;
use Math;
use Config;
use LayerNorm;
use MultiheadAttention;
use DropOut;
use FeedForwardBlock;

class DecoderLayer {
    proc init() {
        norm1 = new LayerNorm();
        mulAtt = new MultiheadAttention();
        dropout1 = new DropOut(batch * sequenceLength * dModel, dropoutRate);
        norm2 = new LayerNorm();
        mulAtt2 = new MultiheadAttention();
        dropout2 = new DropOut(batch * sequenceLength * dModel, dropoutRate);
        norm3 = new LayerNorm();
        pff = new FeedForwardBlock();
        dropout3 = new DropOut(batch * sequenceLength * dModel, dropoutRate);

        domOut = {0..#(batch * sequenceLength * dModel)};
        domGradient = {0..#(batch * sequenceLength * dModel)};
    }

    proc forward(
        ref inputd: [?D] real(32), ref inpute:[D] real(32),
        ref srcSeq: [?Ds] real(32), ref tgtSeq: [Ds] real(32),
        ref output: [D] real(32)) : void {

        norm1.forward(inputd, out1);
        mulAtt1.forward(out1, out1, out1, out2, MaskType.LOOK_AHEAD, tgtSeq);
        dropout1.forward(out2, out3);
        Plus(input, out3, out3);

        norm2.forward(out3, out4);
        mulAtt2.forward(out4, inpute, inpute, out5, MaskType.CROSS_PADDING, srcSeq);
        dropout2.forward(out5, out6);
        Plus(out3, out6, out6);

        norm3.forward(out6, out7);
        pff.forward(out7, out8);
        dropout3.forward(out8, output);
        Plus(out6, output, output);
    }

    proc predict(
        ref inputd: [?D] real(32), ref inpute:[D] real(32),
        ref srcSeq: [?Ds] real(32), ref tgtSeq: [Ds] real(32),
        ref output: [D] real(32)) : void {
            
        norm1.predict(inputd, out1);norm1.updateParameter();
        mulAtt.updateParameter();
        norm2.updateParameter();
        pff.updateParameter();
        mulAtt1.forward(out1, out1, out1, out2, MaskType.LOOK_AHEAD, tgtSeq);
        dropout1.predict(out2, out3);
        Plus(input, out3, out3);

        norm2.predict(out3, out4);
        mulAtt2.forward(out4, inpute, inpute, out5, MaskType.CROSS_PADDING, srcSeq);
        dropout2.predict(out5, out6);
        Plus(out3, out6, out6);

        norm3.predict(out6, out7);
        pff.predict(out7, out8);
        dropout3.predict(out8, output);
        Plus(out6, output, output);
    }

    proc backward(
        ref outputGradient: [?D] real(32), ref inputGradient: [D] real(32),
        ref inputd: [D] real(32), ref inpute:[D] real(32),
        ref srcSeq: [?Ds] real(32), ref tgtSeq: [Ds] real(32),
        ref output: [D] real(32)) : void {

        dropout3.backward(outputGradient, gradient8);
        pff.backward(gradient8, gradient7);
        norm3.backward(gradient7, gradient6);
        Plus(outputGradient, gradient6, gradient6);

        dropout2.backward(gradient6, gradient5);
        mulAtt2.backward(gradient5, gradient4, gradient4, gradient4, MaskType.CROSS_PADDING, srcSeq);
        dropout2.backward(gradient4, gradient3);
        Plus(gradient6, gradient3, gradient3);

        norm1.backward(gradient3, gradient2);
        mulAtt1.backward(gradient2, gradient1);
        dropout1.backward(gradient1, inputGradient);
        Plus(gradient3, inputGradient, inputGradient);
    }

    proc updateParameter() {
        norm1.updateParameter();
        mulAtt.updateParameter();
        norm2.updateParameter();
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

    var domOut: domain(1);
    var domGradient: domain(1);
    var out1: [domOut] real(32);
    var out2: [domOut] real(32);
    var out3: [domOut] real(32);
    var out4: [domOut] real(32);
    var out5: [domOut] real(32);
    var out6: [domOut] real(32);
    var out7: [domOut] real(32);
    var out8: [domOut] real(32);
    var gradient1: [domOut] real(32);
    var gradient2: [domOut] real(32);
    var gradient3: [domOut] real(32);
    var gradient4: [domOut] real(32);
    var gradient5: [domOut] real(32);
    var gradient6: [domOut] real(32);
    var gradient7: [domOut] real(32);
    var gradient8: [domOut] real(32);
}