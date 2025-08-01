use Config;
use Decoder;
use Embedding;
use Encoder;
use Linear;
use Tensor;
use PositionalEncoder;
use Timer;

class Transformer {

    proc init() {
        srcEmbed = new Embedding(srcVocab);
        tgtEmbed = new Embedding(tgtVocab);
        srcPos = new PositionalEncoder();
        tgtPos = new PositionalEncoder();
        decoder = new Decoder();
        encoder = new Encoder();
        linear = new Linear(dModel, tgtVocab);

        domOG = {0..#(batch * sequenceLength * dModel)};
    }

    proc forward(
        ref inpute: [] int, ref inputd: [] int, ref output: [] real(32),
        ref srcSeq: [] int, ref tgtSeq: [] int) : void {

        srcEmbed.forward(inpute, out1);
        srcPos.forward(out1, out2);
        encoder.forward(out2, encoderOut, srcSeq);

        tgtEmbed.forward(inputd, out3);
        tgtPos.forward(out3, out4);
        decoder.forward(out4, encoderOut, out5, srcSeq, tgtSeq);
        linear.forward(out5, output);
    }

    proc predict(
        ref inpute: [] int, ref inputd: [] int, ref output: [] real(32),
        ref srcSeq: [] int, ref tgtSeq: [] int) : void {

        srcEmbed.predict(inpute, out1);
        srcPos.predict(out1, out2);
        encoder.predict(out2, encoderOut, srcSeq);

        tgtEmbed.predict(inputd, out3);
        tgtPos.predict(out3, out4);
        decoder.predict(out4, encoderOut, out5, srcSeq, tgtSeq);
        linear.predict(out5, output);
    }

    proc backward(
        ref outputGradient: [] real(32),
        ref inpute: [] int, ref inputd: [] int,
        ref srcSeq: [] int, ref tgtSeq: [] int) : void {

        linear.backward(outputGradient, gradient5, out5);
        decoder.backward(gradient5, gradient4, encoderGradient, encoderOut, srcSeq, tgtSeq);
        tgtPos.backward(gradient4, gradient3);
        tgtEmbed.backward(gradient3, inputd);
        

        encoder.backward(encoderGradient, gradient2, srcSeq);
        srcPos.backward(gradient2, gradient1);
        srcEmbed.backward(gradient1, inpute);
    }

    proc updateParameter() : void {
        cobegin{
            srcEmbed.updateParameterTask();
            encoder.updateParameterTask();

            tgtEmbed.updateParameterTask();
            decoder.updateParameterTask();
            linear.updateParameterTask();
        }
        
    }

    var srcEmbed: owned Embedding;
    var tgtEmbed: owned Embedding;
    var srcPos: owned PositionalEncoder;
    var tgtPos: owned PositionalEncoder;
    var decoder: owned Decoder;
    var encoder: owned Encoder;
    var linear: owned Linear;

    var domOG: domain(1);
    var encoderOut: [domOG] real(32);
    var encoderGradient: [domOG] real(32);
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
