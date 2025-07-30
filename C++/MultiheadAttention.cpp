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
    MaskType maskType, const int seq[batch], bool train) {

    const int dPerHead = dModel / head;

    SetPar(QT, 0);
    SetPar(KT, 0);
    SetPar(VT, 0);
    SetPar(A, 0);
    SetPar(As, 0);
    SetPar(Ad, 0);
    SetPar(OT, 0);
    SetPar(output, 0);

    for(int i = 0;i < batch;i++) {
        MatMulPlusABTPar(WQ, inputQ.sliceRow(i * sequenceLength, sequenceLength), QT.sliceRow(i * dModel, dModel));
        MatMulPlusABTPar(WK, inputK.sliceRow(i * sequenceLength, sequenceLength), KT.sliceRow(i * dModel, dModel));
        MatMulPlusABTPar(WV, inputV.sliceRow(i * sequenceLength, sequenceLength), VT.sliceRow(i * dModel, dModel));
    }
    #pragma omp parallel for schedule(static)
    for(int i = 0;i < batch;i++) {
        for(int j = 0;j < head;j++) {
            MatMulPlusATBPar(
			QT.sliceRow((i * head + j) * dPerHead, dPerHead), 
			KT.sliceRow((i * head + j) * dPerHead, dPerHead), 
			A.sliceRow((i * head + j) * sequenceLength,sequenceLength));
        }
    }
    DivPar(A, std::sqrt(float(dPerHead)), A);
    #pragma omp parallel for schedule(static)
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
    #pragma omp parallel for schedule(static)
    for(int i = 0;i < batch;i++) {
        for(int j = 0;j < head;j++) {
            MatMulPlusABTPar(
                VT.sliceRow((i * head + j) * dPerHead, dPerHead), 
                Ad.sliceRow((i * head + j) * sequenceLength, sequenceLength), 
                OT.sliceRow((i * head + j) * dPerHead, dPerHead));
        }   
    }
    for(int i = 0;i < batch;i++) {
        MatMulPlusATBPar(OT.sliceRow(i * dModel, dModel), WO, output.sliceRow(i * sequenceLength, sequenceLength));
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
    
    float* WQGrad = WQOpt.gradient.data;
    float* WKGrad = WKOpt.gradient.data;
    float* WVGrad = WVOpt.gradient.data;
    float* WOGrad = WOOpt.gradient.data;

    SetPar(QTGradient, 0);
    SetPar(KTGradient, 0);
    SetPar(VTGradient, 0);
    SetPar(AGradient, 0);
    SetPar(AsGradient, 0);
    SetPar(AdGradient, 0);
    SetPar(OTGradient, 0);
    SetPar(inputGradientQ, 0);
    if(maskType != CROSS_PADDING) { // For cross attention layer
        SetPar(inputGradientK, 0);
        SetPar(inputGradientV, 0);
    }

    for(int i = 0;i < batch;i++) {
        MatMulPlusABPar(OT.sliceRow(i * dModel, dModel),outputGradient.sliceRow(i * sequenceLength, sequenceLength), WOOpt.gradient);
        MatMulPlusABTPar(WO, outputGradient.sliceRow(i * sequenceLength, sequenceLength), OTGradient.sliceRow(i * dModel, dModel));
    }
    #pragma omp parallel for schedule(static)
    for(int i = 0;i < batch;i++) {
        for(int j = 0;j < head;j++) {
            MatMulPlusATBPar(
                OTGradient.sliceRow((i * head + j) * dPerHead, dPerHead),
                VT.sliceRow((i * head + j) * dPerHead, dPerHead),
                AdGradient.sliceRow((i * head + j) * sequenceLength, sequenceLength));
            MatMulPlusABPar(
                OTGradient.sliceRow((i * head + j) * dPerHead, dPerHead),
                Ad.sliceRow((i * head + j) * sequenceLength, sequenceLength),
                VTGradient.sliceRow((i * head + j) * dPerHead, dPerHead));
        }
    }
    Timer::CheckPoint();
    dropout.backward(AdGradient, AsGradient);
    softmax.backward(AsGradient, AGradient, As);
    #pragma omp parallel for schedule(static)
    for(int i = 0;i < batch * head;i++) {
        switch(maskType) {
            case LOOK_AHEAD : ApplyLookAheadMask(AGradient.sliceRow(i * sequenceLength, sequenceLength), seq[i / head], 0); break;
            case PADDING: ApplyPaddingMask(AGradient.sliceRow(i * sequenceLength, sequenceLength), seq[i / head], 0); break;
            case CROSS_PADDING: ApplyCrossPaddingMask(AGradient.sliceRow(i * sequenceLength, sequenceLength), seq[i / head], 0); break;
        }
    }
    DivPar(AGradient, std::sqrt(float(dPerHead)), AGradient);
    #pragma omp parallel for schedule(static)
    for(int i = 0;i < batch;i++) {
        for(int j = 0;j < head;j++) {
            MatMulPlusABPar(
                QT.sliceRow((i * head + j) * dPerHead, dPerHead),
                AGradient.sliceRow((i * head + j) * sequenceLength, sequenceLength),
                KTGradient.sliceRow((i * head + j) * dPerHead, dPerHead));
            MatMulPlusABTPar(
                KT.sliceRow((i * head + j) * dPerHead, dPerHead),
                AGradient.sliceRow((i * head + j) * sequenceLength, sequenceLength),
                QTGradient.sliceRow((i * head + j) * dPerHead, dPerHead));
        }
    }
    for (int i = 0; i < batch; i++) {
        MatMulPlusABPar(QTGradient.sliceRow(i * dModel, dModel), inputQ.sliceRow(i * sequenceLength, sequenceLength), WQOpt.gradient);
        MatMulPlusABPar(KTGradient.sliceRow(i * dModel, dModel), inputK.sliceRow(i * sequenceLength, sequenceLength), WKOpt.gradient);
        MatMulPlusABPar(VTGradient.sliceRow(i * dModel, dModel),  inputV.sliceRow(i * sequenceLength, sequenceLength), WVOpt.gradient);
    }
    for (int i = 0;i < batch;i++) {
        MatMulPlusATBPar(QTGradient.sliceRow(i * dModel, dModel), WQ, inputGradientQ.sliceRow(i * sequenceLength, sequenceLength));
        MatMulPlusATBPar(KTGradient.sliceRow(i * dModel, dModel), WK, inputGradientK.sliceRow(i * sequenceLength, sequenceLength));
        MatMulPlusATBPar(VTGradient.sliceRow(i * dModel, dModel), WV, inputGradientV.sliceRow(i * sequenceLength, sequenceLength));
    }
    Timer::CheckPoint();
}

void MultiheadAttention::updateParameterTask() {
    #pragma omp task
    AdamOpt(WQ, WQOpt);
    #pragma omp task
    AdamOpt(WK, WKOpt);
    #pragma omp task
    AdamOpt(WV, WVOpt);
    #pragma omp task
    AdamOpt(WO, WOOpt);
}

void MultiheadAttention::loadParam(cnpy::npz_t npFile, std::string prefix) {
    WQ.loadNp(npFile, prefix + ".w_q");
    WK.loadNp(npFile, prefix + ".w_k");
    WV.loadNp(npFile, prefix + ".w_v");
    WO.loadNp(npFile, prefix + ".w_o");
}

void MultiheadAttention::checkUpdatedParam(cnpy::npz_t npFile, std::string prefix) {
    Tensor WQUpdated(dModel, dModel);
    Tensor WKUpdated(dModel, dModel);;
    Tensor WVUpdated(dModel, dModel);;
    Tensor WOUpdated(dModel, dModel);;
    WQUpdated.loadNp(npFile, prefix + ".updated_w_q");
    WKUpdated.loadNp(npFile, prefix + ".updated_w_k");
    WVUpdated.loadNp(npFile, prefix + ".updated_w_v");
    WOUpdated.loadNp(npFile, prefix + ".updated_w_o");

    PrintTestResult("backward " + prefix + ".wq", WQ, WQUpdated);
    PrintTestResult("backward " + prefix + ".wk", WK, WKUpdated);
    PrintTestResult("backward " + prefix + ".wv", WV, WVUpdated);
    PrintTestResult("backward " + prefix + ".wo", WO, WOUpdated);
}

void MultiheadAttention::forwardTest(cnpy::npz_t npFile, std::string prefix) {
    Tensor inputQ(batch * sequenceLength, dModel);
    Tensor inputK(batch * sequenceLength, dModel);
    Tensor inputV(batch * sequenceLength, dModel);
    Tensor target(batch * sequenceLength, dModel);
    Tensor output(batch * sequenceLength, dModel);
    Tensor npdLoader(1,1);
    int seq[batch];

    inputQ.loadNp(npFile, prefix + ".q");
    inputK.loadNp(npFile, prefix + ".k");
    inputV.loadNp(npFile, prefix + ".v");
    npdLoader.loadNp(npFile, prefix + ".npd");
    for(int i = 0;i < batch;i++) seq[i] = npdLoader[0];
    target.loadNp(npFile, prefix + ".output");

    forward(inputQ, inputK, inputV, output, LOOK_AHEAD, seq);
    PrintTestResult("forward", output, target);


    
    // Tensor query;
    // Tensor key;
    // Tensor value;
    // Tensor att;
    // Tensor x;
    // query.loadNp(npFile, prefix + ".query");
    // key.loadNp(npFile, prefix + ".key");
    // value.loadNp(npFile, prefix + ".value");
    // att.loadNp(npFile, prefix + ".att");
    // x.loadNp(npFile, prefix + ".x");
    
    // PrintTestResultT("forward query",QT, query);
    // PrintTestResultT("forward key", KT, key);
    // PrintTestResultT("forward value",VT, value);
    // PrintTestResult("forward att",Ad, att);
    // PrintTestResultT("forward out", OT, x);
}

void MultiheadAttention::backwardTest(cnpy::npz_t npFile, std::string prefix) {
    Tensor inputQ(batch * sequenceLength, dModel);
    Tensor inputK(batch * sequenceLength, dModel);
    Tensor inputV(batch * sequenceLength, dModel);
    Tensor target(batch * sequenceLength, dModel);
    Tensor output(batch * sequenceLength, dModel);
    Tensor outputGradient(batch * sequenceLength, dModel);
    Tensor inputGradient(batch * sequenceLength, dModel);
    Tensor npdLoader(1,1);
    int seq[batch];

    outputGradient = 1.0f / outputGradient.row / outputGradient.col;
    inputQ.loadNp(npFile, prefix + ".q");
    inputK.loadNp(npFile, prefix + ".k");
    inputV.loadNp(npFile, prefix + ".v");
    npdLoader.loadNp(npFile, prefix + ".npd");
    for(int i = 0;i < batch;i++) seq[i] = npdLoader[0];

    forward(inputQ, inputK, inputV, output, LOOK_AHEAD, seq);
    backward(outputGradient, inputGradient, inputGradient, inputGradient, inputQ, inputK, inputV, output, LOOK_AHEAD, seq);
    #pragma omp parallel
    {
        #pragma omp single
        {
            updateParameterTask();
            #pragma omp taskwait
        }
    }
    checkUpdatedParam(npFile, prefix);
}