use Util;
use Math;
use Config;
use Softmax;
use Matrix;

enum MaskType {
    LOOK_AHEAD,
    PADDING,
    CROSS_PADDING
}

class MultiheadAttention {
    proc init() {
        softmax = new Softmax(batch * sequenceLength, sequenceLength);
        dropout = new DropOut(batch * sequenceLength, sequenceLength);

        domW = {0..#(dModel & dModel)};
        XavierUniformInit(WQ);
        XavierUniformInit(WK);
        XavierUniformInit(WV);
        XavierUniformInit(WO);

        WQOpt = new AdamOptGradient(WQ);
        WKOpt = new AdamOptGradient(WK);
        WVOpt = new AdamOptGradient(WV);
        WOOpt = new AdamOptGradient(WO);
        
        domO = {0..#(batch * sequenceLength * dModel)};
        domAtt = {0..#(batch * sequenceLength * sequenceLength)};
    }

    proc forward(ref inputQ: [?D] real(32), ref inputK: [D] real(32), ref inputV: [D] real(32), ref output: [D] real(32),
        MaskType: maskType, in seq: int) : void {

        var block: int = dModel * sequenceLength;
        var blockPerHead: int = dPerHead * sequenceLength;
        var blockAtt: int = sequenceLength * sequenceLength;
        
        QT = 0.0;
        KT = 0.0;
        VT = 0.0;
        A = 0.0;
        As = 0.0;
        Ad = 0.0;
        OT = 0.0;
        output = 0.0;

        for i in 0..#batch {
            MatMulPlusABT(WQ, inputQ[(i * block)..#block], QT[(i * block)..#block]);
            MatMulPlusABT(WK, inputK[(i * block)..#block], KT[(i * block)..#block]);
            MatMulPlusABT(WV, inputV[(i * block)..#block], VT[(i * block)..#block]);
        }
        for i in 0..#(batch * head) {
            MatMulPlusATB(QT[(i * blockPerHead)..#blockPerHead], KT[(i * blockPerHead)..#blockPerHead], A[(i * blockAtt)..#blockAtt]);
        }
        Div(A, sqrt(dPerHead), A);
        select maskType {
            when MaskType.LOOK_AHEAD do ApplyLookAheadMask(A, seq, -1e9);
            when MaskType.PADDING do ApplyPaddingMask(A, seq, -1e9);
            when MaskType.CROSS_PADDING do ApplyCrossPaddingMask(A, seq, -1e9);
        }
        softmax.forward(A, As);
        dropout.forward(As, Ad);
        forall i in 0..#(batch * head) {
            MatMulPlusABT(VT[(i * blockPerHead)..#blockPerHead], Ad[(i * blockAtt)..#blockAtt], OT[(i * blockPerHead)..#blockPerHead]);
        }
        forall i in 0..#batch {
            MatMulPlusATB(OT[(i * block)..#block], WO, output[(i * block)..#block]);
        }
    }

    proc predict(ref inputQ: [?D] real(32), ref inputK: [D] real(32), ref inputV: [D] real(32), ref output: [D] real(32),
        MaskType: maskType, in seq: int) : void {
        
        var block: int = dModel * sequenceLength;
        var blockPerHead: int = dPerHead * sequenceLength;
        var blockAtt: int = sequenceLength * sequenceLength;
        
        QT = 0.0;
        KT = 0.0;
        VT = 0.0;
        A = 0.0;
        As = 0.0;
        Ad = 0.0;
        OT = 0.0;
        output = 0.0;

        for i in 0..#batch {
            MatMulPlusABT(WQ, inputQ[(i * block)..#block], QT[(i * block)..#block]);
            MatMulPlusABT(WK, inputK[(i * block)..#block], KT[(i * block)..#block]);
            MatMulPlusABT(WV, inputV[(i * block)..#block], VT[(i * block)..#block]);
        }
        for i in 0..#(batch * head) {
            MatMulPlusATB(QT[(i * blockPerHead)..#blockPerHead], KT[(i * blockPerHead)..#blockPerHead], A[(i * blockAtt)..#blockAtt]);
        }
        Div(A, sqrt(dPerHead), A);
        select maskType {
            when MaskType.LOOK_AHEAD do ApplyLookAheadMask(A, seq, -1e9);
            when MaskType.PADDING do ApplyPaddingMask(A, seq, -1e9);
            when MaskType.CROSS_PADDING do ApplyCrossPaddingMask(A, seq, -1e9);
        }
        softmax.predict(A, As);
        dropout.predict(As, Ad);
        forall i in 0..#(batch * head) {
            MatMulPlusABT(VT[(i * blockPerHead)..#blockPerHead], Ad[(i * blockAtt)..#blockAtt], OT[(i * blockPerHead)..#blockPerHead]);
        }
        forall i in 0..#batch {
            MatMulPlusATB(OT[(i * block)..#block], WO, output[(i * block)..#block]);
        }
    }

    proc backward(ref outputGradient: real(32), ref inputGradientQ: real(32), ref inputGradientK: real(32), ref inputGradientV: real(32),
        ref inputQ: [?D] real(32), ref inputK: [D] real(32), ref inputV: [D] real(32), ref output: [D] real(32),
        MaskType: maskType, in seq: int) {
        
        var block: int = dModel * sequenceLength;
        var blockPerHead: int = dPerHead * sequenceLength;
        var blockAtt: int = sequenceLength * sequenceLength;

        QTGradient = 0.0;
        KTGradient = 0.0;
        VTGradient = 0.0;
        AGradient = 0.0;
        AsGradient = 0.0;
        AdGradient = 0.0;
        OTGradient = 0.0;

        for i in 0..#batch {
            MatMulPlusAB(OT[(i * block)..#block], outputGradient[(i * block)..#block], WOOpt.gradient);
            MatMulPlusABT(WO, outputGradient[(i * block)..#block], OTGradient[(i * block)..#block]);
        }
        for i in 0..#(batch * head) {
            MatMulPlusATB(OTGradient[(i * blockPerHead)..#blockPerHead], VT[(i * blockPerHead)..#blockPerHead], AdGradient[(i * blockAtt)..#blockAtt]);
            MatMulPlusAB(OTGradient[(i * blockPerHead)..#blockPerHead], Ad[(i * blockAtt)..#blockAtt], VTGradient[(i * blockPerHead)..#blockPerHead]);
        }
        dropout.backward(AdGradient, AsGradient);
        dropout.backward(AsGradient, AGradient);
        Div(AGradient, sqrt(dPerHead), AGradient);
        for i in 0..#(batch * head) {
            MatMulPlusABT(KT[(i * blockPerHead)..#blockPerHead], AGradient[(i * blockAtt)..#blockAtt], QTGradient[(i * blockPerHead)..#blockPerHead]);
            MatMulPlusAB(QT[(i * blockPerHead)..#blockPerHead], AGradient[(i * blockAtt)..#blockAtt], KTGradient[(i * blockPerHead)..#blockPerHead]);
        }
        for i in 0..#batch {
            MatMulPlusAB(QTGradient[(i * block)..#block], inputQ[(i * block)..#block], WQOpt.gradient);
            MatMulPlusAB(KTGradient[(i * block)..#block], inputK[(i * block)..#block], WKOpt.gradient);
            MatMulPlusAB(VTGradient[(i * block)..#block], inputV[(i * block)..#block], WVOpt.gradient);
            MatMulPlusATB(QTGradient[(i * block)..#block], WQ, inputGradientQ[(i * block)..#block]);
            MatMulPlusATB(KTGradient[(i * block)..#block], WK, inputGradientK[(i * block)..#block]);
            MatMulPlusATB(VTGradient[(i * block)..#block], WV, inputGradientV[(i * block)..#block]);
        }
    }

    proc updateParameter() {
        AdamOpt(WQ, WQOpt);
        AdamOpt(WK, WKOpt);
        AdamOpt(WV, WVOpt);
        AdamOpt(WO, WOOpt);
    }

    var softmax: owned Softmax;
    var dropout: owned DropOut;

    var domW: domain(1);
    var WQ: [domW] real(32);
    var WK: [domW] real(32);
    var WV: [domW] real(32);
    var WO: [domW] real(32);

    var WQOpt: AdamOptGradient2;
    var WKOpt: AdamOptGradient2;
    var WVOpt: AdamOptGradient2;
    var WOOpt: AdamOptGradient2;

    var domO: domain(1);
    var domAtt: domain(1);
    var QT: [domO] real(32);
    var KT: [domO] real(32);
    var VT: [domO] real(32);
    var A: [domAtt] real(32);
    var As: [domAtt] real(32);
    var Ad: [domAtt] real(32);
    var OT: [domO] real(32);
}