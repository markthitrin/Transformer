#ifndef MULTIHEAD_ATTENTION
#define MULTIHEAD_ATTENTION

#include "Header.h"
#include "Tensor.h"
#include "Util.h"
#include "Softmax.h"

template<int head,int d,int l,int in,int q,int out>
class MultiheadAttention {
public:
	MultiheadAttention() noexcept :
		_WQ			(Create<q, in>()),
		_WK			(Create<q, in>()),
		_WV			(Create<out, in>()),
		_WO			(Create<out, out>()),

		_WQGradient	(Create0<q, in>()),
		_WQM		(Create0<q, in>()),
		_WQV		(Create0<q, in>()),
		_WKGradient	(Create0<q, in>()),
		_WKM		(Create0<q, in>()),
		_WKV		(Create0<q, in>()),
		_WVGradient	(Create0<out, in>()),
		_WVM		(Create0<out, in>()),
		_WVV		(Create0<out, in>()),
		_WOGradient	(Create0<out, out>()),
		_WOM		(Create0<out, out>()),
		_WOV		(Create0<out, out>()),

		_QT			(Create<d * q, l>()),
		_KT			(Create<d * q, l>()),
		_VT			(Create<d * out, l>()),
		_A			(Create<d * head * l, l>()),
		_As			(Create<d * head * l, l>()),
		_OT			(Create<d * out, l>()),

		_QTGradient	(Create<d * q, l>()),
		_KTGradient	(Create<d * q, l>()),
		_VTGradient	(Create<d * out, l>()),
		_AGradient	(Create<d * head * l, l>()),
		_AsGradient	(Create<d * head * l, l>()),
		_OTGradient	(Create<d * out, l>()) {

		XavierUniformInit<q, in>(_WQ);
		XavierUniformInit<q, in>(_WK);
		XavierUniformInit<out, in>(_WV);
		XavierUniformInit<out, out>(_WO);

		softmax._input = _A;
		softmax._output = _As;
		softmax._inGradient = _AsGradient;
		softmax._outGradient = _AGradient;
	}
	~MultiheadAttention() {
		std::free(_WQ);
		std::free(_WK);
		std::free(_WV);
		std::free(_WO);

		std::free(_WQGradient);
		std::free(_WQM);
		std::free(_WQV);
		std::free(_WKGradient);
		std::free(_WKM);
		std::free(_WKV);
		std::free(_WVGradient);
		std::free(_WVM);
		std::free(_WVV);
		std::free(_WOGradient);
		std::free(_WOM);
		std::free(_WOV);

		std::free(_QT);
		std::free(_KT);
		std::free(_VT);
		std::free(_A);
		std::free(_As);
		std::free(_OT);

		std::free(_QTGradient);
		std::free(_KTGradient);
		std::free(_VTGradient);
		std::free(_AGradient);
		std::free(_AsGradient);
		std::free(_OTGradient);
	}

	void forward() noexcept {
		IMPORT_CONST(inputQ);
		IMPORT_CONST(inputK);
		IMPORT_CONST(inputV);
		IMPORT_CONST(WQ);
		IMPORT_CONST(WK);
		IMPORT_CONST(WV);
		IMPORT_CONST(WO);
		IMPORT(QT);
		IMPORT(KT);
		IMPORT(VT);
		IMPORT(A);
		IMPORT(As);
		IMPORT(OT);
		IMPORT(output);
		constexpr int qPerHead = q / head;
		constexpr int outPerHead = out / head;
		Reset<d * q, l>(QT);
		Reset<d * q, l>(KT);
		Reset<d * out, l>(VT);
		Reset<d * head * l, l>(A);
		Reset<d * head * l, l>(As);
		Reset<d * out, l>(OT);
		Reset<d * out, l>(output);
		for (int i = 0; i < d; i++) {
			MatMulPlusABT<q, in, l>(Tensor(WQ), Tensor(inputQ + i * (l * in)), Tensor(QT + i * (l * q)));
			MatMulPlusABT<q, in, l>(Tensor(WK), Tensor(inputK + i * (l * in)), Tensor(KT + i * (l * q)));
			MatMulPlusABT<out, in, l>(Tensor(WV), Tensor(inputV + i * (l * in)), Tensor(VT + i * (l * out)));
		}
		for (int i = 0; i < d; i++) {
			for (int j = 0; j < head; j++) {
				const int ij = (i * head + j);
				MatMulPlusATB<l, qPerHead, l>(Tensor(QT + ij * (qPerHead * l)), Tensor(KT + ij * (qPerHead * l)), Tensor(A + ij * (l * l)));
			}
		}
		Div<d * head * l, l>(Tensor(A), std::sqrt(float(qPerHead)), Tensor(A));
		ApplyLookAheadMask<d * head, l, -1e9f>(Tensor(A));
		softmax.forward();
		for (int i = 0; i < d; i++) {
			for (int j = 0; j < head; j++) {
				const int ij = (i * head + j);
				MatMulPlusABT<outPerHead, l, l>(Tensor(VT + ij * (outPerHead * l)), Tensor(As + ij * (l * l)), Tensor(OT + ij * (outPerHead * l)));
			}
		}
		for (int i = 0; i < d; i++) {
			MatMulPlusATB<l, out, out>(Tensor(OT + i * (l * out)), Tensor(WO), Tensor(output + i * (l * out)));
		}
	}

	void predict() noexcept {
		forward();
	}

	void backpropagate() noexcept {
		IMPORT_CONST(inGradient);
		IMPORT_CONST(inputQ);
		IMPORT_CONST(inputK);
		IMPORT_CONST(inputV);
		IMPORT_CONST(WQ);
		IMPORT_CONST(WK);
		IMPORT_CONST(WV);
		IMPORT_CONST(WO);
		IMPORT_CONST(QT);
		IMPORT_CONST(KT);
		IMPORT_CONST(VT);
		IMPORT_CONST(A);
		IMPORT_CONST(As);
		IMPORT_CONST(OT);
		IMPORT_CONST(output);
		IMPORT(outGradientQ);
		IMPORT(outGradientK);
		IMPORT(outGradientV);
		IMPORT(WQGradient);
		IMPORT(WQM);
		IMPORT(WQV);
		IMPORT(WKGradient);
		IMPORT(WKM);
		IMPORT(WKV);
		IMPORT(WVGradient);
		IMPORT(WVM);
		IMPORT(WVV);
		IMPORT(WOGradient);
		IMPORT(WOM);
		IMPORT(WOV);
		IMPORT(QTGradient);
		IMPORT(KTGradient);
		IMPORT(VTGradient);
		IMPORT(AGradient);
		IMPORT(AsGradient);
		IMPORT(OTGradient);
		Reset<d* q, l>(QTGradient);
		Reset<d* q, l>(KTGradient);
		Reset<d* out, l>(VTGradient);
		Reset<d* head* l, l>(AGradient);
		Reset<d* head* l, l>(AsGradient);
		Reset<d* out, l>(OTGradient);
		Reset<d* in, l>(outGradientQ);
		Reset<d* in, l>(outGradientK);
		Reset<d* out, l>(outGradientV);
		feedCount++;
		constexpr int qPerHead = q / head;
		constexpr int outPerHead = out / head;
		for (int i = 0; i < d; i++) {
			MatMulPlusAB<out, l, out>(Tensor(OT + i * (out * l)), Tensor(inGradient + i * (l * out)), Tensor(WOGradient));
			MatMulPlusABT<out, out, l>(Tensor(WO), Tensor(inGradient + i * (out * l)), Tensor(OTGradient + i * (out * l)));
		}
		for (int i = 0; i < d; i++) {
			for (int j = 0; j < head; j++) {
				const int ij = (i * head + j);
				MatMulPlusATB<l, outPerHead, l>(Tensor(OTGradient + ij * (outPerHead * l)), Tensor(VT + ij * (outPerHead * l)), Tensor(AsGradient + ij * (l * l)));
				MatMulPlusAB<outPerHead, l, l>(Tensor(OTGradient + ij * (outPerHead * l)), Tensor(As + ij * (l * l)), Tensor(VTGradient + ij * (outPerHead * l)));
			}
		}
		softmax.backpropagate();
		ApplyLookAheadMask<d* head, l, 0f>((Tensor)AGradient);
		Div<d* head* l, l>((Tensor)AGradient, std::sqrt(float(qPerHead)), (Tensor)AGradient);
		for (int i = 0; i < d; i++) {
			for (int j = 0; j < head; j++) {
				const int ij = (i * head + j);
				MatMulPlusAB<qPerHead, l, l>(Tensor(QT + ij * (qPerHead * l)), Tensor(AGradient + ij * (l * l)), Tensor(KTGradient + ij * (qPerHead * l)));
				MatMulPlusABT<qPerHead, l, l>(Tensor(KT + ij * (qPerHead * l)), Tensor(AGradient + ij * (l * l)), Tensor(QTGradient + ij * (qPerHead * l)));
			}
		}

		for (int i = 0; i < d; i++) {
			MatMulPlusAB<q, l, in>(Tensor(QTGradient + i * (q * l)), Tensor(inputQ + i * (l * in)), Tensor(WQGradient));
			MatMulPlusAB<q, l, in>(Tensor(KTGradient + i * (q * l)), Tensor(inputK + i * (l * in)), Tensor(WKGradient));
			MatMulPlusAB<out, l, in>(Tensor(VTGradient + i * (out * l)), Tensor(inputV + i * (l * in)), Tensor(WVGradient));
			MatMulPlusATB<l, q, in>(Tensor(QTGradient + i * (q * l)), Tensor(WQ), Tensor(outGradientQ + i * (l * in)));
			MatMulPlusATB<l, q, in>(Tensor(KTGradient + i * (q * l)), Tensor(WK), Tensor(outGradientK + i * (l * in)));
			MatMulPlusATB<l, out, in>(Tensor(VTGradient + i * (out * l)), Tensor(WV), Tensor(outGradientV + i * (l * in)));
		}
	}

	void updateParameter() noexcept {
		IMPORT(WQ);
		IMPORT(WK);
		IMPORT(WV);
		IMPORT(WO);
		IMPORT(WQGradient);
		IMPORT(WQM);
		IMPORT(WQV);
		IMPORT(WKGradient);
		IMPORT(WKM);
		IMPORT(WKV);
		IMPORT(WVGradient);
		IMPORT(WVM);
		IMPORT(WVV);
		IMPORT(WOGradient);
		IMPORT(WOM);
		IMPORT(WOV);
		Div<q, in>(WQGradient, feedCount, WQGradient);
		Div<q, in>(WKGradient, feedCount, WKGradient);
		Div<out, in>(WVGradient, feedCount, WVGradient);
		Div<out, out>(WOGradient, feedCount, WOGradient);

		AdamOpt<q, in>(WQ, WQM, WQV, WQGradient, t);
		AdamOpt<q, in>(WK, WKM, WKV, WKGradient, t);
		AdamOpt<out, in>(WV, WVM, WVV, WVGradient, t);
		AdamOpt<out, out>(WO, WOM, WOV, WOGradient, t);

		Reset<q, in>(WQGradient);
		Reset<q, in>(WKGradient);
		Reset<out, in>(WVGradient);
		Reset<out, out>(WOGradient);
		feedCount = 0;
		t++;
	}

	Tensor _inputQ;
	Tensor _inputK;
	Tensor _inputV;
	Tensor _output;
	Tensor _inGradient;
	Tensor _outGradientQ;
	Tensor _outGradientK;
	Tensor _outGradientV;

	Tensor _WQ;
	Tensor _WK;
	Tensor _WV;
	Tensor _WO;

	int t = 1;
	int feedCount = 0;
	Tensor _WQGradient;
	Tensor _WQM;
	Tensor _WQV;
	Tensor _WKGradient;
	Tensor _WKM;
	Tensor _WKV;
	Tensor _WVGradient;
	Tensor _WVM;
	Tensor _WVV;
	Tensor _WOGradient;
	Tensor _WOM;
	Tensor _WOV;

	Tensor _QT;
	Tensor _KT;
	Tensor _VT;
	Tensor _A;
	Tensor _As;
	Tensor _OT;

	Tensor _QTGradient;
	Tensor _KTGradient;
	Tensor _VTGradient;
	Tensor _AGradient;
	Tensor _AsGradient;
	Tensor _OTGradient;

	Softmax<d* head, l> softmax;
};

#endif // !MULTIHEAD_ATTENTION
