use Util;
use Math;
use Config;
use Softmax;
use Matrix;
use DropOut;
use Timer;

enum MaskType {
    LOOK_AHEAD,
    PADDING,
    CROSS_PADDING
}

class MultiheadAttention {
    proc init() {
        softmax = new Softmax(batch * head * sequenceLength, sequenceLength);
        dropout = new DropOut(batch * head * sequenceLength * sequenceLength);

        domW = {0..#(dModel * dModel)};
        WQ = 0;
        WK = 0;
        WV = 0;
        WO = 0;
        XavierUniformInit(WQ);
        XavierUniformInit(WK);
        XavierUniformInit(WV);
        XavierUniformInit(WO);

        WQOpt = new AdamOptimizer(WQ);
        WKOpt = new AdamOptimizer(WK);
        WVOpt = new AdamOptimizer(WV);
        WOOpt = new AdamOptimizer(WO);
        
        domO = {0..#(batch * sequenceLength * dModel)};
        domAtt = {0..#(batch * head * sequenceLength * sequenceLength)};
    }

    proc process(ref inputQ: [?D] real(32), ref inputK: [D] real(32), ref inputV: [D] real(32), ref output: [D] real(32),
        maskType: MaskType, in seq: [?Ds] int, in train: bool) : void {
        
        var dPerHead: int = dModel / head;
        var block: int = dModel * sequenceLength;
        var blockPerHead: int = dPerHead * sequenceLength;
        var blockAtt: int = sequenceLength * sequenceLength;
        
        Set(QT, 0.0);
        Set(KT, 0.0);
        Set(VT, 0.0);
        Set(A, 0.0);
        Set(As, 0.0);
        Set(Ad, 0.0);
        Set(OT, 0.0);
        Set(output, 0.0);

        for i in 0..#batch {
            MatMulPlusABT(dModel, dModel, sequenceLength, WQ, inputQ[(i * block)..#block], QT[(i * block)..#block]);
            MatMulPlusABT(dModel, dModel, sequenceLength, WK, inputK[(i * block)..#block], KT[(i * block)..#block]);
            MatMulPlusABT(dModel, dModel, sequenceLength, WV, inputV[(i * block)..#block], VT[(i * block)..#block]);
        }
        for i in 0..#(batch * head) {
            MatMulPlusATB(sequenceLength, dPerHead, sequenceLength, QT[(i * blockPerHead)..#blockPerHead], KT[(i * blockPerHead)..#blockPerHead], A[(i * blockAtt)..#blockAtt]);
        }
        Div(A, sqrt(dPerHead):real(32), A);
        for i in 0..#(batch * head) {
            select maskType {
                when MaskType.LOOK_AHEAD do ApplyLookAheadMask(A[(i * blockAtt)..#blockAtt], seq[i / head], -1e9);
                when MaskType.PADDING do ApplyPaddingMask(A[(i * blockAtt)..#blockAtt], seq[i / head], -1e9);
                when MaskType.CROSS_PADDING do ApplyCrossPaddingMask(A[(i * blockAtt)..#blockAtt], seq[i / head], -1e9);
            }
        }
        CheckPoint();
        if(train) {
            softmax.forward(A, As);
            dropout.forward(As, Ad);
        }
        else {
            softmax.predict(A, As);
            dropout.predict(As, Ad);
        }
        forall i in 0..#(batch * head) {
            MatMulPlusABT(dPerHead, sequenceLength, sequenceLength, VT[(i * blockPerHead)..#blockPerHead], Ad[(i * blockAtt)..#blockAtt], OT[(i * blockPerHead)..#blockPerHead]);
        }
        CheckPoint();
        forall i in 0..#batch {
            MatMulPlusATB(sequenceLength, dModel, dModel, OT[(i * block)..#block], WO, output[(i * block)..#block]);
        }
    }

    proc forward(ref inputQ: [?D] real(32), ref inputK: [D] real(32), ref inputV: [D] real(32), ref output: [D] real(32),
        maskType: MaskType, in seq: [?Ds] int) : void {
        
        process(inputQ, inputK, inputV, output, maskType, seq, true);
    }

    proc predict(ref inputQ: [?D] real(32), ref inputK: [D] real(32), ref inputV: [D] real(32), ref output: [D] real(32),
        maskType: MaskType, in seq: [?Ds] int) : void {
        
        process(inputQ, inputK, inputV, output, maskType, seq, false);
    }

    proc backward(ref outputGradient: [?D] real(32), ref inputGradientQ: [D] real(32), ref inputGradientK: [D] real(32), ref inputGradientV: [D] real(32),
        ref inputQ: [D] real(32), ref inputK: [D] real(32), ref inputV: [D] real(32), ref output: [D] real(32),
        maskType: MaskType, in seq: [?Ds] int) {
        
        var dPerHead: int = dModel / head;
        var block: int = dModel * sequenceLength;
        var blockPerHead: int = dPerHead * sequenceLength;
        var blockAtt: int = sequenceLength * sequenceLength;

        Set(QTGradient, 0.0);
        Set(KTGradient, 0.0);
        Set(VTGradient, 0.0);
        Set(AGradient, 0.0);
        Set(AsGradient, 0.0);
        Set(AdGradient, 0.0);
        Set(OTGradient, 0.0);
        Set(inputGradientQ, 0.0);
        if maskType != MaskType.CROSS_PADDING {
            Set(inputGradientK, 0.0);
            Set(inputGradientV, 0.0);
        }

        for i in 0..#batch {
            MatMulPlusAB(dModel, sequenceLength, dModel, OT[(i * block)..#block], outputGradient[(i * block)..#block], WOOpt.gradient);
            MatMulPlusABT(dModel, dModel, sequenceLength, WO, outputGradient[(i * block)..#block], OTGradient[(i * block)..#block]);
        }
        for i in 0..#(batch * head) {
            MatMulPlusATB(sequenceLength, dPerHead, sequenceLength, OTGradient[(i * blockPerHead)..#blockPerHead], VT[(i * blockPerHead)..#blockPerHead], AdGradient[(i * blockAtt)..#blockAtt]);
            MatMulPlusAB(dPerHead, sequenceLength, sequenceLength, OTGradient[(i * blockPerHead)..#blockPerHead], Ad[(i * blockAtt)..#blockAtt], VTGradient[(i * blockPerHead)..#blockPerHead]);
        }
        dropout.backward(AdGradient, AsGradient);
        softmax.backward(AsGradient, AGradient, As);
        for i in 0..#(batch * head) {
            select maskType {
                when MaskType.LOOK_AHEAD do ApplyLookAheadMask(AGradient[(i * blockAtt)..#blockAtt], seq[i / head], 0);
                when MaskType.PADDING do ApplyPaddingMask(AGradient[(i * blockAtt)..#blockAtt], seq[i / head], 0);
                when MaskType.CROSS_PADDING do ApplyCrossPaddingMask(AGradient[(i * blockAtt)..#blockAtt], seq[i / head], 0);
            }
        }
        Div(AGradient, sqrt(dPerHead):real(32), AGradient);
        for i in 0..#(batch * head) {
            MatMulPlusABT(dPerHead, sequenceLength, sequenceLength, KT[(i * blockPerHead)..#blockPerHead], AGradient[(i * blockAtt)..#blockAtt], QTGradient[(i * blockPerHead)..#blockPerHead]);
            MatMulPlusAB(dPerHead, sequenceLength, sequenceLength, QT[(i * blockPerHead)..#blockPerHead], AGradient[(i * blockAtt)..#blockAtt], KTGradient[(i * blockPerHead)..#blockPerHead]);
        }
        for i in 0..#batch {
            MatMulPlusAB(dModel, sequenceLength, dModel, QTGradient[(i * block)..#block], inputQ[(i * block)..#block], WQOpt.gradient);
            MatMulPlusAB(dModel, sequenceLength, dModel, KTGradient[(i * block)..#block], inputK[(i * block)..#block], WKOpt.gradient);
            MatMulPlusAB(dModel, sequenceLength, dModel, VTGradient[(i * block)..#block], inputV[(i * block)..#block], WVOpt.gradient);
            MatMulPlusATB(sequenceLength, dModel, dModel, QTGradient[(i * block)..#block], WQ, inputGradientQ[(i * block)..#block]);
            MatMulPlusATB(sequenceLength, dModel, dModel, KTGradient[(i * block)..#block], WK, inputGradientK[(i * block)..#block]);
            MatMulPlusATB(sequenceLength, dModel, dModel, VTGradient[(i * block)..#block], WV, inputGradientV[(i * block)..#block]);
        }
    }

    proc updateParameter() {
        AdamOpt(WQ, WQOpt);
        AdamOpt(WK, WKOpt);
        AdamOpt(WV, WVOpt);
        AdamOpt(WO, WOOpt);
    }

    proc loadParam() {
        loadM(WQ);
        loadM(WK);
        loadM(WV);
        loadM(WO);
    }

    proc forwardTest() {
        var inputQ: [0..#(batch * sequenceLength * dModel)] real(32);
        var inputK: [0..#(batch * sequenceLength * dModel)] real(32);
        var inputV: [0..#(batch * sequenceLength * dModel)] real(32);
        var output: [0..#(batch * sequenceLength * dModel)] real(32);
        var target: [0..#(batch * sequenceLength * dModel)] real(32);
        var npdLoader: [0..#1] real(32);
        var seq: [0..#batch] int;

        loadM(inputQ);
        loadM(inputK);
        loadM(inputV);
        loadM(npdLoader);
        loadM(target);
        for i in 0..#batch do seq[i] = npdLoader[0]:int;

        forward(inputQ, inputK, inputV, output, MaskType.LOOK_AHEAD, seq);

        PrintTestResult("forward", output, target);
    }

    proc checkUpdateParam() {
        var WQUpdated: [0..#(dModel * dModel)] real(32);
        var WKUpdated: [0..#(dModel * dModel)] real(32);
        var WVUpdated: [0..#(dModel * dModel)] real(32);
        var WOUpdated: [0..#(dModel * dModel)] real(32);

        loadM(WQUpdated);
        loadM(WKUpdated);
        loadM(WVUpdated);
        loadM(WOUpdated);
        
        PrintTestResult("backward wQ", WQ, WQUpdated);
        PrintTestResult("backward wK", WK, WKUpdated);
        PrintTestResult("backward wV", WV, WVUpdated);
        PrintTestResult("backward wO", WO, WOUpdated);
    }

    proc backwardTest() {
        var inputQ: [0..#(batch * sequenceLength * dModel)] real(32);
        var inputK: [0..#(batch * sequenceLength * dModel)] real(32);
        var inputV: [0..#(batch * sequenceLength * dModel)] real(32);
        var output: [0..#(batch * sequenceLength * dModel)] real(32);
        var target: [0..#(batch * sequenceLength * dModel)] real(32);
        var outputGradient: [0..#(batch * sequenceLength * dModel)] real(32);
        var inputGradient: [0..#(batch * sequenceLength * dModel)] real(32);
        var npdLoader: [0..#1] real(32);
        var seq: [0..#batch] int;
        
        outputGradient = (1.0 / outputGradient.domain.size):real(32);

        loadM(inputQ);
        loadM(inputK);
        loadM(inputV);
        loadM(npdLoader);
        for i in 0..#batch do seq[i] = npdLoader[0]:int;

        forward(inputQ, inputK, inputV, output, MaskType.LOOK_AHEAD, seq);
        backward(
            outputGradient, inputGradient, inputGradient, inputGradient,
            inputQ, inputK, inputV, output,
            MaskType.LOOK_AHEAD, seq);
        updateParameter();

        checkUpdateParam();
    }

    var softmax: owned Softmax;
    var dropout: owned DropOut;

    var domW: domain(1);
    var WQ: [domW] real(32);
    var WK: [domW] real(32);
    var WV: [domW] real(32);
    var WO: [domW] real(32);

    var WQOpt: AdamOptimizer;
    var WKOpt: AdamOptimizer;
    var WVOpt: AdamOptimizer;
    var WOOpt: AdamOptimizer;

    var domO: domain(1);
    var domAtt: domain(1);
    var QT: [domO] real(32);
    var KT: [domO] real(32);
    var VT: [domO] real(32);
    var A: [domAtt] real(32);
    var As: [domAtt] real(32);
    var Ad: [domAtt] real(32);
    var OT: [domO] real(32);

    var QTGradient: [domO] real(32);
    var KTGradient: [domO] real(32);
    var VTGradient: [domO] real(32);
    var AGradient: [domAtt] real(32);
    var AsGradient: [domAtt] real(32);
    var AdGradient: [domAtt] real(32);
    var OTGradient: [domO] real(32);
}

// Test code
// var model = new MultiheadAttention();
// model.loadParam();
// for i in 0..4 do model.forwardTest();
// for i in 0..4 do model.backwardTest();