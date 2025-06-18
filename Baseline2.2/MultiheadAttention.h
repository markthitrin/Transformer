#ifndef MULTIHEAD_ATTENTION
#define MULTIHEAD_ATTENTION

#include "Header.h"
#include "Tensor.h"
#include "Util.h"
#include "Softmax.h"
#include "DropOut.h"

enum MaskType {
	LOOK_AHEAD,
	PADDING,
	CROSS_PADDING
};

template<int head, int maskType, int batch,int len,int col>
class MultiheadAttention {
public:
	MultiheadAttention() {
		_WQ.XavierUniformInit();
		_WK.XavierUniformInit();
		_WV.XavierUniformInit();
		_WO.XavierUniformInit();

		_QT.init();
		_KT.init();
		_VT.init();
		_A.init();
		_As.init();
		_OT.init();

		_QTGradient.init();
		_KTGradient.init();
		_VTGradient.init();
		_AGradient.init();
		_AsGradient.init();
		_OTGradient.init();

		softmax._input = _A;
		softmax._output = _As;
		softmax._inGradient = _AsGradient;
		softmax._outGradient = _AGradient;

		dropout._input = _As;
		dropout._output = _As;
		dropout._inGradient = _AsGradient;
		dropout._outGradient = _AsGradient;
	}
	~MultiheadAttention() {
		
	}

	void forward(int npd[batch]) noexcept {
		constexpr int colPerhead = col / head;

		Reset(_QT);
		Reset(_KT);
		Reset(_VT);
		Reset(_A);
		Reset(_As);
		Reset(_OT);
		Reset(_output);
		for (int i = 0; i < batch; i++) {
			MatMulPlusABT(_WQ, _inputQ.template sliceRow<len>(i * len), _QT.template sliceRow<col>(i * col));
			MatMulPlusABT(_WK, _inputK.template sliceRow<len>(i * len), _KT.template sliceRow<col>(i * col));
			MatMulPlusABT(_WV, _inputV.template sliceRow<len>(i * len), _VT.template sliceRow<col>(i * col));
		}
		for (int i = 0; i < batch * head; i++) {
			MatMulPlusATB(
				_QT.template sliceRow<colPerhead>(i * colPerhead), 
				_KT.template sliceRow<colPerhead>(i * colPerhead), 
				_A.template sliceRow<len>(i * len));
		}
		Div(_A, std::sqrt(float(colPerhead)), _A);
		switch(maskType) {
			case LOOK_AHEAD :
				for(int i = 0;i < batch * head;i++) { 
					ApplyLookAheadMask<len>(_A.template sliceRow<len>(i * len), npd[i / head], -1e9);
				} break;
			case PADDING: 
				for(int i = 0;i < batch * head;i++) {
					ApplyPaddingMask<len>(_A.template sliceRow<len>(i * len), npd[i / head], -1e9);
				} break;
			case CROSS_PADDING:
				for(int i = 0;i < batch * head;i++) {
					ApplyCrossPaddingMask<len>(_A.template sliceRow<len>(i * len), npd[i / head], -1e9);
				} break;
		}
		softmax.forward();
		dropout.forward();
		for (int i = 0; i < batch * head; i++) {
			MatMulPlusABT(
				_VT.template sliceRow<colPerhead>(i * colPerhead), 
				_As.template sliceRow<len>(i * len), 
				_OT.template sliceRow<colPerhead>(i * colPerhead));
		}
		for (int i = 0; i < batch; i++) {
			MatMulPlusATB(_OT.template sliceRow<col>(i * col), _WO, _output.template sliceRow<len>(i * len));
		}
	}

	void predict(int npd[batch]) noexcept {
		forward(npd);
	}

	void backward(int npd[batch]) noexcept {
		feedCount++;
		constexpr int colPerhead = col / head;

		Reset(_QTGradient);
		Reset(_KTGradient);
		Reset(_VTGradient);
		Reset(_AGradient);
		Reset(_AsGradient);
		Reset(_OTGradient);
		Reset(_outGradientQ);
		Reset(_outGradientK);
		Reset(_outGradientV);
		for (int i = 0; i < batch; i++) {
			MatMulPlusAB(_OT.template sliceRow<col>(i * col),_inGradient.template sliceRow<len>(i * len), _WOOpt.gradient);
			MatMulPlusABT(_WO, _inGradient.template sliceRow<len>(i * len), _OTGradient.template sliceRow<col>(i * col));
		}
		for (int i = 0; i < batch * head; i++) {
			MatMulPlusATB(
				_OTGradient.template sliceRow<colPerhead>(i * colPerhead),
				_VT.template sliceRow<colPerhead>(i * colPerhead),
				_AsGradient.template sliceRow<len>(i * len));
			MatMulPlusAB(
				_OTGradient.template sliceRow<colPerhead>(i * colPerhead),
				_As.template sliceRow<len>(i * len),
				_VTGradient.template sliceRow<colPerhead>(i * colPerhead));
		}
		dropout.backward();
		softmax.backward();
		switch(maskType) {
			case LOOK_AHEAD :
				for(int i = 0;i < batch * head;i++) { 
					ApplyLookAheadMask<len>(_A.template sliceRow<len>(i * len), npd[i / head], 0);
				} break;
			case PADDING: 
				for(int i = 0;i < batch * head;i++) {
					ApplyPaddingMask<len>(_A.template sliceRow<len>(i * len), npd[i / head], 0);
				} break;
			case CROSS_PADDING:
				for(int i = 0;i < batch * head;i++) {
					ApplyCrossPaddingMask<len>(_A.template sliceRow<len>(i * len), npd[i / head], 0);
				} break;
		}
		Div(_AGradient, std::sqrt(float(colPerhead)), _AGradient);
		for (int i = 0; i < batch * head; i++) {
			MatMulPlusAB(
				_QT.template sliceRow<colPerhead>(i * colPerhead),
				_AGradient.template sliceRow<len>(i * len),
				_KTGradient.template sliceRow<colPerhead>(i * colPerhead));
			MatMulPlusABT(
				_KT.template sliceRow<colPerhead>(i * colPerhead),
				_AGradient.template sliceRow<len>(i * len),
				_QTGradient.template sliceRow<colPerhead>(i * colPerhead));
		}

		for (int i = 0; i < batch; i++) {
			MatMulPlusAB(_QTGradient.template sliceRow<col>(i * col), _inputQ.template sliceRow<len>(i * len), _WQOpt.gradient);
			MatMulPlusAB(_KTGradient.template sliceRow<col>(i * col), _inputK.template sliceRow<len>(i * len), _WKOpt.gradient);
			MatMulPlusAB(_VTGradient.template sliceRow<col>(i * col),  _inputV.template sliceRow<len>(i * len), _WVOpt.gradient);
			MatMulPlusATB(_QTGradient.template sliceRow<col>(i * col), _WQ, _outGradientQ.template sliceRow<len>(i * len));
			MatMulPlusATB(_KTGradient.template sliceRow<col>(i * col), _WK, _outGradientK.template sliceRow<len>(i * len));
			MatMulPlusATB(_VTGradient.template sliceRow<col>(i * col), _WV, _outGradientV.template sliceRow<len>(i * len));
		}
	}

	void updateParameter() noexcept {
		AdamOpt(_WQ, _WQOpt, feedCount);
		AdamOpt(_WK, _WKOpt, feedCount);
		AdamOpt(_WV, _WVOpt, feedCount);
		AdamOpt(_WO, _WOOpt, feedCount);

		feedCount = 0;
	}

	Tensor<batch * len, col> _inputQ;
	Tensor<batch * len, col> _inputK;
	Tensor<batch * len, col> _inputV;
	Tensor<batch * len, col> _output;
	Tensor<batch * len, col> _inGradient;
	Tensor<batch * len, col> _outGradientQ;
	Tensor<batch * len, col> _outGradientK;
	Tensor<batch * len, col> _outGradientV;

	Tensor<col, col> _WQ;
	Tensor<col, col> _WK;
	Tensor<col, col> _WV;
	Tensor<col, col> _WO;

	int feedCount = 0;
	AdamOptGradient<col, col> _WQOpt;
	AdamOptGradient<col, col> _WKOpt;
	AdamOptGradient<col, col> _WVOpt;
	AdamOptGradient<col, col> _WOOpt;

	Tensor<batch * col, len> _QT;
	Tensor<batch * col, len> _KT;
	Tensor<batch * col, len> _VT;
	Tensor<batch * head * len, len> _A;
	Tensor<batch * head * len, len>  _As;
	Tensor<batch * col, len>  _OT;

	Tensor<batch * col, len>  _QTGradient;
	Tensor<batch * col, len>  _KTGradient;
	Tensor<batch * col, len>  _VTGradient;
	Tensor<batch * head * len, len> _AGradient;
	Tensor<batch * head * len, len> _AsGradient;
	Tensor<batch * col, len>  _OTGradient;

	Softmax<batch * head * len, len> softmax;
	DropOut<batch * head * len, len, dropoutRate> dropout;
};

#endif // !MULTIHEAD_ATTENTION
