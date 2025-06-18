#ifndef TRANSFORMER
#define TRANSFORMER

#include "Header.h"
#include "Tensor.h"
#include "Encoder.h"
#include "Softmax.h"
#include "PositionalEncoder.h"
#include "Embedding.h"
#include "Linear.h"
#include "LayerNorm.h"
#include "Util.h"
#include "Decoder.h"

class Transformer {
public:
	Transformer() noexcept :
        _inputEncoder(srcEmbed._input),
        _inputDecoder(tgtEmbed._input),
        _output(linear._output),
        _inGradient(linear._inGradient) {

        _encoderOut.init();
        _encoderGradient.init(); 
        _out1.init();
        _out2.init();
        _out3.init();
        _out4.init();
        _out5.init();
        _gradient1.init();
        _gradient2.init();
        _gradient3.init();
        _gradient4.init();
        _gradient5.init();


        srcPos._input = srcEmbed._output = _out1;
        srcEmbed._inGradient = srcPos._outGradient = _gradient1;
        encoder._input = srcPos._output = _out2;
        srcPos._inGradient = encoder._outGradient = _gradient2;


        tgtPos._input = tgtEmbed._output = _out3;
        tgtEmbed._inGradient = tgtPos._outGradient = _gradient3;
        decoder._input = tgtPos._output = _out4;
        tgtPos._inGradient = decoder._outGradient = _gradient4;
        
        encoder._output = _encoderOut;
        encoder._inGradient = _encoderGradient;
        decoder.setEncodePtrTo(_encoderOut, _encoderGradient);

        linear._input = decoder._output = _out5;
        decoder._inGradient = linear._outGradient = _gradient5;
	}
	~Transformer() {
        _encoderOut.free();
        _encoderGradient.free();
		_out1.free();
		_out2.free();
		_out3.free();
		_out4.free();
		_out5.free();
        _gradient1.free();
        _gradient2.free();
        _gradient3.free();
        _gradient4.free();
        _gradient5.free();
	}

    void forward(int npdSrc[batch], int npdTgt[batch]) noexcept {
        srcEmbed.forward();
        srcPos.forward();
        encoder.forward(npdSrc);
        tgtEmbed.forward();
        tgtPos.forward();
        decoder.forward(npdSrc, npdTgt);
        linear.forward();
    }

	void predict(int npdSrc[batch], int npdTgt[batch]) noexcept {
		srcEmbed.predict();
        srcPos.predict();
        encoder.predict(npdSrc);
        tgtEmbed.predict();
        tgtPos.predict();
        decoder.predict(npdSrc, npdTgt);
        linear.predict();
	}

	void backward(int npdSrc[batch], int npdTgt[batch]) noexcept {
		linear.backward();
        decoder.backward(npdSrc, npdTgt);
        tgtPos.backward();
        tgtEmbed.backward();
        encoder.backward(npdSrc);
        srcPos.backward();
        srcEmbed.backward();
	}

	void updateParameter() noexcept {
        srcEmbed.updateParameter();
        encoder.updateParameter();
        tgtEmbed.updateParameter();
        decoder.updateParameter();
        linear.updateParameter();
	}

    Embedding<batch * sequenceLength, srcVocab, dModel> srcEmbed;
    Embedding<batch * sequenceLength, tgtVocab, dModel> tgtEmbed;
    PositionalEncoder<batch, sequenceLength, dModel> srcPos;
    PositionalEncoder<batch, sequenceLength, dModel> tgtPos;
    Decoder<batch, sequenceLength, dModel, N> decoder;
    Encoder<batch, sequenceLength, dModel, N> encoder;
    Linear<batch * sequenceLength, dModel, tgtVocab> linear;

	Tensor<1, batch * sequenceLength>& _inputEncoder;
    Tensor<1, batch * sequenceLength>& _inputDecoder;
	Tensor<batch * sequenceLength, tgtVocab>& _output;
	Tensor<batch * sequenceLength, tgtVocab>& _inGradient;

	Tensor<batch * sequenceLength, dModel> _encoderOut;
    Tensor<batch * sequenceLength, dModel> _encoderGradient;
    Tensor<batch * sequenceLength, dModel> _out1;
    Tensor<batch * sequenceLength, dModel> _out2;
    Tensor<batch * sequenceLength, dModel> _out3;
    Tensor<batch * sequenceLength, dModel> _out4;
    Tensor<batch * sequenceLength, dModel> _out5;
    Tensor<batch * sequenceLength, dModel> _gradient1;
    Tensor<batch * sequenceLength, dModel> _gradient2;
    Tensor<batch * sequenceLength, dModel> _gradient3;
    Tensor<batch * sequenceLength, dModel> _gradient4;
    Tensor<batch * sequenceLength, dModel> _gradient5;
};

#endif // !TRANSFORMER
