#ifndef MULTIHEAD_ATTENTION
#define MULTIHEAD_ATTENTION

#include "Header.cuh"
#include "Tensor.cuh"
#include "Softmax.cuh"
#include "DropOut.cuh"

enum MaskType {
	LOOK_AHEAD,
	PADDING,
	CROSS_PADDING
};

class MultiheadAttention {
public:
	MultiheadAttention(
		Tensor inputQ,
		Tensor inputK,
		Tensor inputV,
		Tensor output,
		Tensor outputGradient,
		Tensor inputGradientQ,
		Tensor inputGradientK,
		Tensor inputGradientV) noexcept:
		softmax(_A, _As, _AsGradient, _AGradient),
		dropout(_As, _Ad, _AdGradient, _AsGradient),
		_inputQ(inputQ),
		_inputK(inputK),
		_inputV(inputV),
		_output(output),
		_outputGradient(outputGradient),
		_inputGradientQ(inputGradientQ),
		_inputGradientK(inputGradientK),
		_inputGradientV(inputGradientV) {

		_WQ.XavierUniformInit();
		_WK.XavierUniformInit();
		_WV.XavierUniformInit();
		_WO.XavierUniformInit();
	}
	~MultiheadAttention() {
		_WQ.free();
		_WK.free();
		_WV.free();
		_WO.free();

		_QT.free();
		_KT.free();
		_VT.free();
		_A.free();
		_As.free();
		_Ad.free();
		_OT.free();

		_QTGradient.free();
		_KTGradient.free();
		_VTGradient.free();
		_AGradient.free();
		_AsGradient.free();
		_AdGradient.free();
		_OTGradient.free();
	}

	void forward(int npd) noexcept {
		constexpr int colPerhead = col / head;

		Reset(_QT);
		Reset(_KT);
		Reset(_VT);
		Reset(_A);
		Reset(_Ad);
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
			case LOOK_AHEAD : ApplyLookAheadMask<batch * head, len>(_A, npd, -1e9); break;
			case PADDING: ApplyPaddingMask<batch * head, len>(_A, npd, -1e9); break;
			case CROSS_PADDING: ApplyCrossPaddingMask<batch * head, len>(_A, npd, -1e9); break;
		}
		softmax.forward();
		dropout.forward();
		for (int i = 0; i < batch * head; i++) {
			MatMulPlusABT(
				_VT.template sliceRow<colPerhead>(i * colPerhead), 
				_Ad.template sliceRow<len>(i * len), 
				_OT.template sliceRow<colPerhead>(i * colPerhead));
		}
		for (int i = 0; i < batch; i++) {
			MatMulPlusATB(_OT.template sliceRow<col>(i * col), _WO, _output.template sliceRow<len>(i * len));
		}
	}

	void predict(int npd) noexcept {
		forward(npd);
	}

	void backward(int npd) noexcept {
		feedCount++;
		constexpr int colPerhead = col / head;

		Reset(_QTGradient);
		Reset(_KTGradient);
		Reset(_VTGradient);
		Reset(_AGradient);
		Reset(_AdGradient);
		Reset(_OTGradient);
		Reset(_inputGradientQ);
		Reset(_inputGradientK);
		Reset(_inputGradientV);
		for (int i = 0; i < batch; i++) {
			MatMulPlusAB(_OT.template sliceRow<col>(i * col),_outputGradient.template sliceRow<len>(i * len), _WOOpt.gradient);
			MatMulPlusABT(_WO, _outputGradient.template sliceRow<len>(i * len), _OTGradient.template sliceRow<col>(i * col));
		}
		for (int i = 0; i < batch * head; i++) {
			MatMulPlusATB(
				_OTGradient.template sliceRow<colPerhead>(i * colPerhead),
				_VT.template sliceRow<colPerhead>(i * colPerhead),
				_AdGradient.template sliceRow<len>(i * len));
			MatMulPlusAB(
				_OTGradient.template sliceRow<colPerhead>(i * colPerhead),
				_Ad.template sliceRow<len>(i * len),
				_VTGradient.template sliceRow<colPerhead>(i * colPerhead));
		}
		dropout.backward();
		softmax.backward();
		switch(maskType) {
			case LOOK_AHEAD : ApplyLookAheadMask<batch * head, len>(_AGradient, npd, 0); break;
			case PADDING: ApplyPaddingMask<batch * head, len>(_AGradient, npd, 0); break;
			case CROSS_PADDING: ApplyCrossPaddingMask<batch * head, len>(_AGradient, npd, 0); break;
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
			MatMulPlusATB(_QTGradient.template sliceRow<col>(i * col), _WQ, _inputGradientQ.template sliceRow<len>(i * len));
			MatMulPlusATB(_KTGradient.template sliceRow<col>(i * col), _WK, _inputGradientK.template sliceRow<len>(i * len));
			MatMulPlusATB(_VTGradient.template sliceRow<col>(i * col), _WV, _inputGradientV.template sliceRow<len>(i * len));
		}
	}

	void updateParameter() noexcept {
		AdamOpt(_WQ, _WQOpt, feedCount);
		AdamOpt(_WK, _WKOpt, feedCount);
		AdamOpt(_WV, _WVOpt, feedCount);
		AdamOpt(_WO, _WOOpt, feedCount);

		feedCount = 0;
	}

	void loadParam(cnpy::npz_t npFile, std::string prefix) {
		_WQ.loadNp(npFile, prefix + ".w_q");
		_WK.loadNp(npFile, prefix + ".w_k");
		_WV.loadNp(npFile, prefix + ".w_v");
		_WO.loadNp(npFile, prefix + ".w_o");
		for(int i = 0;i  < col;i++) {
			for(int j = 0;j < i;j++) {
				std::swap(_WO.data[i*col + j],_WO.data[j*col + i]);
			}
		}
	}

	void checkUpdatedParam(cnpy::npz_t npFile, std::string prefix) {
		Tensor WQUpdated;
		Tensor WKUpdated;
		Tensor WVUpdated;
		Tensor WOUpdated;
		WQUpdated.loadNp(npFile, prefix + ".original_w_q");
		WKUpdated.loadNp(npFile, prefix + ".original_w_k");
		WVUpdated.loadNp(npFile, prefix + ".original_w_v");
		WOUpdated.loadNp(npFile, prefix + ".original_w_o");
		for(int i = 0;i  < col;i++) {
			for(int j = 0;j < i;j++) {
				std::swap(WOUpdated.data[i*col + j],WOUpdated.data[j*col + i]);
			}
		}
		PrintTestResult("backward " + prefix + ".wq", _WQ, WQUpdated);
		PrintTestResult("backward " + prefix + ".wk", _WK, WKUpdated);
		PrintTestResult("backward " + prefix + ".wv", _WV, WVUpdated);
		PrintTestResult("backward " + prefix + ".wo", _WO, WOUpdated);
	}

	void forwardTest(cnpy::npz_t npFile, std::string prefix) {
		Tensor<batch * sequenceLength, col> target;
		Tensor<1, 1> npdLoader;

		_inputQ.loadNp(npFile, prefix + ".q");
		_inputK.loadNp(npFile, prefix + ".k");
		_inputV.loadNp(npFile, prefix + ".v");
		npdLoader.loadNp(npFile, prefix + ".npd");
		target.loadNp(npFile, prefix + ".output");

		forward(npdLoader.data[0]);
		PrintTestResult("forward",_output, target);


		
		// Tensor<batch * sequenceLength, col> query;
		// Tensor<batch * sequenceLength, col> key;
		// Tensor<batch * sequenceLength, col> value;
		// Tensor<batch * head * sequenceLength, sequenceLength> att;
		// Tensor<batch * sequenceLength, col> x;
		// query.loadNp(npFile, prefix + ".query");
		// key.loadNp(npFile, prefix + ".key");
		// value.loadNp(npFile, prefix + ".value");
		// att.loadNp(npFile, prefix + ".att");
		// x.loadNp(npFile, prefix + ".x");
		
		// PrintTestResultT<batch, col, sequenceLength>("forward query",_QT, query);
		// PrintTestResultT<batch, col, sequenceLength>("forward key", _KT, key);
		// PrintTestResultT<batch, col, sequenceLength>("forward value",_VT, value);
		// PrintTestResult("forward att",_Ad, att);
		// PrintTestResultT<batch, col, sequenceLength>("forward out", _OT, x);
	}

	void backwardTest(cnpy::npz_t npFile, std::string prefix) {
		Set(_outputGradient, 1.0f / sequenceLength / col);
		_inputQ.loadNp(npFile, prefix + ".q");
		_inputK.loadNp(npFile, prefix + ".k");
		_inputV.loadNp(npFile, prefix + ".v");
		Tensor<1, 1> npdLoader;
		npdLoader.loadNp(npFile, prefix + ".npd");

		forward(npdLoader.data[0]);
		backward(npdLoader.data[0]);
		updateParameter();

		checkUpdatedParam(npFile, prefix);
	}

	Tensor inputQ;
	Tensor inputK;
	Tensor inputV;
	Tensor output;
	Tensor outputGradient;
	Tensor inputGradientQ;
	Tensor inputGradientK;
	Tensor inputGradientV;

	Tensor WQ;
	Tensor WK;
	Tensor WV;
	Tensor WO;

	int feedCount = 0;
	AdamOptimizer WQOpt;
	AdamOptGradient WKOpt;
	AdamOptGradient WVOpt;
	AdamOptGradient WOOpt;

	Tensor QT;
	Tensor KT;
	Tensor VT;
	Tensor A;
	Tensor As;
	Tensor Ad;
	Tensor OT;

	Tensor QTGradient;
	Tensor KTGradient;
	Tensor VTGradient;
	Tensor AGradient;
	Tensor AsGradient;
	Tensor AdGradient;
	Tensor OTGradient;

	Softmax softmax;
	DropOut dropout;
};

