use Config;
use DropOut;
use Math;
use Tensor;
use Softmax;
use Timer;
use Util;

enum MaskType {
    LOOK_AHEAD,
    PADDING,
    CROSS_PADDING
}

class MultiheadAttention {
    proc init() {
        softmax = new Softmax();
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

    proc process(
        ref inputQ: [] real(32), ref inputK: [] real(32), ref inputV: [] real(32), ref output: [] real(32),
        maskType: MaskType, in seq: [] int, in train: bool) : void {
        
        var dPerHead: int = dModel / head;
        var block: int = dModel * sequenceLength;
        var blockPerHead: int = dPerHead * sequenceLength;
        var blockAtt: int = sequenceLength * sequenceLength;
        
        SetPar(0, batch * dModel * sequenceLength, QT, 0.0);
        SetPar(0, batch * dModel * sequenceLength, KT, 0.0);
        SetPar(0, batch * dModel * sequenceLength, VT, 0.0);
        SetPar(0, batch * head * sequenceLength * sequenceLength, A, 0.0);
        SetPar(0, batch * head * sequenceLength * sequenceLength, As, 0.0);
        SetPar(0, batch * head * sequenceLength * sequenceLength, Ad, 0.0);
        SetPar(0, batch * dModel * sequenceLength, OT, 0.0);
        SetPar(0, batch * sequenceLength * dModel, output, 0.0);

        for i in 0..#batch {
            MatMulPlusABTPar(dModel, dModel, sequenceLength, WQ, inputQ[(i * block)..#block], QT[(i * block)..#block]);
            MatMulPlusABTPar(dModel, dModel, sequenceLength, WK, inputK[(i * block)..#block], KT[(i * block)..#block]);
            MatMulPlusABTPar(dModel, dModel, sequenceLength, WV, inputV[(i * block)..#block], VT[(i * block)..#block]);
        }
        forall i in 0..#batch {
            for j in (i * head)..#head {
                MatMulPlusATBPar(sequenceLength, dPerHead, sequenceLength, QT[(j * blockPerHead)..#blockPerHead], KT[(j * blockPerHead)..#blockPerHead], A[(j * blockAtt)..#blockAtt]);
            }
        }
        DivPar(0, 0, batch * head * sequenceLength * sequenceLength, A, sqrt(dPerHead):real(32), A);
        forall i in 0..#(batch * head) {
            select maskType {
                when MaskType.LOOK_AHEAD do ApplyLookAheadMask(i * blockAtt, A, seq[i / head], -1e9);
                when MaskType.PADDING do ApplyPaddingMask(i * blockAtt, A, seq[i / head], -1e9);
                when MaskType.CROSS_PADDING do ApplyCrossPaddingMask(i * blockAtt, A, seq[i / head], -1e9);
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
        forall i in 0..#batch {
            for j in (i * head)..#head {
                MatMulPlusABTPar(dPerHead, sequenceLength, sequenceLength, VT[(j * blockPerHead)..#blockPerHead], Ad[(j * blockAtt)..#blockAtt], OT[(j * blockPerHead)..#blockPerHead]);
            }
        }
        for i in 0..#batch {
            MatMulPlusATBPar(sequenceLength, dModel, dModel, OT[(i * block)..#block], WO, output[(i * block)..#block]);
        }
        CheckPoint();
    }

    proc forward(
        ref inputQ: [] real(32), ref inputK: [] real(32), ref inputV: [] real(32), ref output: [] real(32),
        maskType: MaskType, in seq: [?Ds] int) : void {
        
        process(inputQ, inputK, inputV, output, maskType, seq, true);
    }

    proc predict(
        ref inputQ: [] real(32), ref inputK: [] real(32), ref inputV: [] real(32), ref output: [] real(32),
        maskType: MaskType, in seq: [?Ds] int) : void {
        
        process(inputQ, inputK, inputV, output, maskType, seq, false);
    }

    proc backward(
        ref outputGradient: [] real(32), ref inputGradientQ: [] real(32), ref inputGradientK: [] real(32), ref inputGradientV: [] real(32),
        ref inputQ: [] real(32), ref inputK: [] real(32), ref inputV: [] real(32), ref output: [] real(32),
        maskType: MaskType, in seq: [] int) : void {
        
        var dPerHead: int = dModel / head;
        var block: int = dModel * sequenceLength;
        var blockPerHead: int = dPerHead * sequenceLength;
        var blockAtt: int = sequenceLength * sequenceLength;

        SetPar(0, batch * dModel * sequenceLength, QTGradient, 0.0);
        SetPar(0, batch * dModel * sequenceLength, KTGradient, 0.0);
        SetPar(0, batch * dModel * sequenceLength, VTGradient, 0.0);
        SetPar(0, batch * head * sequenceLength * sequenceLength, AGradient, 0.0);
        SetPar(0, batch * head * sequenceLength * sequenceLength, AsGradient, 0.0);
        SetPar(0, batch * head * sequenceLength * sequenceLength, AdGradient, 0.0);
        SetPar(0, batch * dModel * sequenceLength, OTGradient, 0.0);
        SetPar(0, batch * sequenceLength * dModel, inputGradientQ, 0.0);
        if maskType != MaskType.CROSS_PADDING { // For cross attention layer
            SetPar(0, batch * sequenceLength * dModel, inputGradientK, 0.0);
            SetPar(0, batch * sequenceLength * dModel, inputGradientV, 0.0);
        }

        for i in 0..#batch {
            MatMulPlusABPar(dModel, sequenceLength, dModel, OT[(i * block)..#block], outputGradient[(i * block)..#block], WOOpt.gradient);
            MatMulPlusABTPar(dModel, dModel, sequenceLength, WO, outputGradient[(i * block)..#block], OTGradient[(i * block)..#block]);
        }
        forall i in 0..#batch {
            for j in (i * head)..#head {
                MatMulPlusATBPar(sequenceLength, dPerHead, sequenceLength, OTGradient[(j * blockPerHead)..#blockPerHead], VT[(j * blockPerHead)..#blockPerHead], AdGradient[(j * blockAtt)..#blockAtt]);
                MatMulPlusABPar(dPerHead, sequenceLength, sequenceLength, OTGradient[(j * blockPerHead)..#blockPerHead], Ad[(j * blockAtt)..#blockAtt], VTGradient[(j * blockPerHead)..#blockPerHead]);
            }
        }
        CheckPoint();
        dropout.backward(AdGradient, AsGradient);
        softmax.backward(AsGradient, AGradient, As);
        forall i in 0..#(batch * head) {
            select maskType {
                when MaskType.LOOK_AHEAD do ApplyLookAheadMask(i * blockAtt, AGradient, seq[i / head], 0);
                when MaskType.PADDING do ApplyPaddingMask(i * blockAtt, AGradient, seq[i / head], 0);
                when MaskType.CROSS_PADDING do ApplyCrossPaddingMask(i * blockAtt, AGradient, seq[i / head], 0);
            }
        }
        DivPar(0, 0, domAtt.size, AGradient, sqrt(dPerHead):real(32), AGradient);
        forall i in 0..#batch {
            for j in (i * head)..#head {
                MatMulPlusABTPar(dPerHead, sequenceLength, sequenceLength, KT[(j * blockPerHead)..#blockPerHead], AGradient[(j * blockAtt)..#blockAtt], QTGradient[(j * blockPerHead)..#blockPerHead]);
                MatMulPlusABPar(dPerHead, sequenceLength, sequenceLength, QT[(j * blockPerHead)..#blockPerHead], AGradient[(j * blockAtt)..#blockAtt], KTGradient[(j * blockPerHead)..#blockPerHead]);
            }
        }
        for i in 0..#batch {
            MatMulPlusABPar(dModel, sequenceLength, dModel, QTGradient[(i * block)..#block], inputQ[(i * block)..#block], WQOpt.gradient);
            MatMulPlusABPar(dModel, sequenceLength, dModel, KTGradient[(i * block)..#block], inputK[(i * block)..#block], WKOpt.gradient);
            MatMulPlusABPar(dModel, sequenceLength, dModel, VTGradient[(i * block)..#block], inputV[(i * block)..#block], WVOpt.gradient);
        }
        for i in 0..#batch {
            MatMulPlusATBPar(sequenceLength, dModel, dModel, QTGradient[(i * block)..#block], WQ, inputGradientQ[(i * block)..#block]);
            MatMulPlusATBPar(sequenceLength, dModel, dModel, KTGradient[(i * block)..#block], WK, inputGradientK[(i * block)..#block]);
            MatMulPlusATBPar(sequenceLength, dModel, dModel, VTGradient[(i * block)..#block], WV, inputGradientV[(i * block)..#block]);
        }
        CheckPoint();
    }

    proc updateParameterTask() : void {
        cobegin {
            AdamOpt(WQ, WQOpt);
            AdamOpt(WK, WKOpt);
            AdamOpt(WV, WVOpt);
            AdamOpt(WO, WOOpt);
        }
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
        updateParameterTask();

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