#include "Config.h"
#include "DropOut.h"
#include "Header.h"
#include "MultiheadAttention.h"
#include "Softmax.h"
#include "Tensor.h"
#include "Timer.h"
#include "Util.h"

MultiheadAttention::MultiheadAttention() :
    softmax(),
    dropout(batch * head * sequenceLength, sequenceLength),

    WQ(dModel, dModel),
    WK(dModel, dModel),
    WV(dModel, dModel),
    WO(dModel, dModel),

    WQOpt(dModel, dModel),
    WKOpt(dModel, dModel),
    WVOpt(dModel, dModel),
    WOOpt(dModel, dModel),

    QT(batch * dModel, sequenceLength),
	KT(batch * dModel, sequenceLength),
	VT(batch * dModel, sequenceLength),
	A(batch * head * sequenceLength, sequenceLength),
	As(batch * head * sequenceLength, sequenceLength),
    Ad(batch * head * sequenceLength, sequenceLength),
	OT(batch * dModel, sequenceLength),

	QTGradient(batch * dModel, sequenceLength),
	KTGradient(batch * dModel, sequenceLength),
	VTGradient(batch * dModel, sequenceLength),
	AGradient(batch * head * sequenceLength, sequenceLength),
	AsGradient(batch * head * sequenceLength, sequenceLength),
	AdGradient(batch * head * sequenceLength, sequenceLength),
	OTGradient(batch * dModel, sequenceLength) {

    XavierUniformInit(WQ);
    XavierUniformInit(WK);
    XavierUniformInit(WV);
    XavierUniformInit(WO);
}

void MultiheadAttention::process(
    TensorView inputQ, TensorView inputK, TensorView inputV, TensorView output,
    MaskType maskType, const  int seq[batch], bool train) {

    const int dPerHead = dModel / head;

    QT = 0;
    KT = 0;
    VT = 0;
    A = 0;
    As = 0;
    Ad = 0;
    OT = 0;
    output = 0;

    for(int i = 0;i < batch;i++) {
        MatMulPlusABT(WQ, inputQ.sliceRow(i * sequenceLength, sequenceLength), QT.sliceRow(i * dModel, dModel));
        MatMulPlusABT(WK, inputK.sliceRow(i * sequenceLength, sequenceLength), KT.sliceRow(i * dModel, dModel));
        MatMulPlusABT(WV, inputV.sliceRow(i * sequenceLength, sequenceLength), VT.sliceRow(i * dModel, dModel));
    }
    for(int i = 0;i < batch * head;i++) {
        MatMulPlusATB(
			QT.sliceRow(i * dPerHead, dPerHead), 
			KT.sliceRow(i * dPerHead, dPerHead), 
			A.sliceRow(i * sequenceLength,sequenceLength));
    }
    Div(A, std::sqrt(float(dPerHead)), A);
    for(int i = 0;i < batch * head;i++) {
        switch(maskType) {
            case LOOK_AHEAD : ApplyLookAheadMask(A.sliceRow(i * sequenceLength, sequenceLength), seq[i / head], -1e9); break;
            case PADDING: ApplyPaddingMask(A.sliceRow(i * sequenceLength, sequenceLength), seq[i / head], -1e9); break;
            case CROSS_PADDING: ApplyCrossPaddingMask(A.sliceRow(i * sequenceLength, sequenceLength), seq[i / head], -1e9); break;
        }
    }
    Timer::CheckPoint();
    if(train) {
        softmax.forward(A, As);
        dropout.forward(As, Ad);
    }
    else {
        softmax.predict(A, As);
        dropout.predict(As, Ad);
    }
    for (int i = 0; i < batch * head; i++) {
        MatMulPlusABT(
            VT.sliceRow(i * dPerHead, dPerHead), 
            Ad.sliceRow(i * sequenceLength, sequenceLength), 
            OT.sliceRow(i * dPerHead, dPerHead));
    }
    for (int i = 0; i < batch; i++) {
        MatMulPlusATB(OT.sliceRow(i * dModel, dModel), WO, output.sliceRow(i * sequenceLength, sequenceLength));
    }
    Timer::CheckPoint();
}

void MultiheadAttention::forward(
    TensorView inputQ, TensorView inputK, TensorView inputV, TensorView output,
    MaskType maskType, const int seq[batch]) {

    process(inputQ, inputK, inputV, output, maskType, seq, true);
}

void MultiheadAttention::predict(
    TensorView inputQ, TensorView inputK, TensorView inputV, TensorView output,
    MaskType maskType, const int seq[batch]) {

    process(inputQ, inputK, inputV, output, maskType, seq, false);
}

void MultiheadAttention::backward(
    TensorView outputGradient, TensorView inputGradientQ, TensorView inputGradientK, TensorView inputGradientV,
    TensorView inputQ, TensorView inputK, TensorView inputV, TensorView output,
    MaskType maskType, const int seq[batch]) {

    const int dPerHead = dModel / head;

    QTGradient = 0;
    KTGradient = 0;
    VTGradient = 0;
    AGradient = 0;
    AsGradient = 0;
    AdGradient = 0;
    OTGradient = 0;
    inputGradientQ = 0;
    if(maskType != CROSS_PADDING) { // for decoder getting encoder layer
        inputGradientK = 0;
        inputGradientV = 0;
    }

    for (int i = 0; i < batch; i++) {
        MatMulPlusAB(OT.sliceRow(i * dModel, dModel),outputGradient.sliceRow(i * sequenceLength, sequenceLength), WOOpt.gradient);
        MatMulPlusABT(WO, outputGradient.sliceRow(i * sequenceLength, sequenceLength), OTGradient.sliceRow(i * dModel, dModel));
    }
    for (int i = 0; i < batch * head; i++) {
        MatMulPlusATB(
            OTGradient.sliceRow(i * dPerHead, dPerHead),
            VT.sliceRow(i * dPerHead, dPerHead),
            AdGradient.sliceRow(i * sequenceLength, sequenceLength));
        MatMulPlusAB(
            OTGradient.sliceRow(i * dPerHead, dPerHead),
            Ad.sliceRow(i * sequenceLength, sequenceLength),
            VTGradient.sliceRow(i * dPerHead, dPerHead));
    }
    Timer::CheckPoint();
    dropout.backward(AdGradient, AsGradient);
    softmax.backward(AsGradient, AGradient, As);
    for(int i = 0;i < batch * head;i++) {
        switch(maskType) {
            case LOOK_AHEAD : ApplyLookAheadMask(AGradient.sliceRow(i * sequenceLength, sequenceLength), seq[i / head], 0); break;
            case PADDING: ApplyPaddingMask(AGradient.sliceRow(i * sequenceLength, sequenceLength), seq[i / head], 0); break;
            case CROSS_PADDING: ApplyCrossPaddingMask(AGradient.sliceRow(i * sequenceLength, sequenceLength), seq[i / head], 0); break;
        }
    }
    Div(AGradient, std::sqrt(float(dPerHead)), AGradient);
    
    for (int i = 0; i < batch * head; i++) {
        MatMulPlusAB(
            QT.sliceRow(i * dPerHead, dPerHead),
            AGradient.sliceRow(i * sequenceLength, sequenceLength),
            KTGradient.sliceRow(i * dPerHead, dPerHead));
        MatMulPlusABT(
            KT.sliceRow(i * dPerHead, dPerHead),
            AGradient.sliceRow(i * sequenceLength, sequenceLength),
            QTGradient.sliceRow(i * dPerHead, dPerHead));
    }
    for (int i = 0; i < batch; i++) {
        MatMulPlusAB(QTGradient.sliceRow(i * dModel, dModel), inputQ.sliceRow(i * sequenceLength, sequenceLength), WQOpt.gradient);
        MatMulPlusAB(KTGradient.sliceRow(i * dModel, dModel), inputK.sliceRow(i * sequenceLength, sequenceLength), WKOpt.gradient);
        MatMulPlusAB(VTGradient.sliceRow(i * dModel, dModel),  inputV.sliceRow(i * sequenceLength, sequenceLength), WVOpt.gradient);
    }
    for (int i = 0;i < batch;i++) {
        MatMulPlusATB(QTGradient.sliceRow(i * dModel, dModel), WQ, inputGradientQ.sliceRow(i * sequenceLength, sequenceLength));
        MatMulPlusATB(KTGradient.sliceRow(i * dModel, dModel), WK, inputGradientK.sliceRow(i * sequenceLength, sequenceLength));
        MatMulPlusATB(VTGradient.sliceRow(i * dModel, dModel), WV, inputGradientV.sliceRow(i * sequenceLength, sequenceLength));
    }
    Timer::CheckPoint();
}

void MultiheadAttention::updateParameter() {
    AdamOpt(WQ, WQOpt);
    AdamOpt(WK, WKOpt);
    AdamOpt(WV, WVOpt);
    AdamOpt(WO, WOOpt);
}