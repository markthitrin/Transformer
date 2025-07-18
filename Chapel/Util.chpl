use Config;
use Math;
use Timer;

record AdamOptimizer {
    proc init(ref parameter: [?D] real(32)) {
        dom = D;
        t = 1;
    }

    proc init(in dom: domain(1)) {
        this.dom = dom;
        t = 1;
    }
    
    var dom: domain(1);
    var gradient: [dom] real(32);
    var accM: [dom] real(32);
    var accV: [dom] real(32);
    var t: int;
};

proc AdamOpt(ref parameter: [?D] real(32), ref opt: AdamOptimizer) : void {
    var learningRate:real(32) = lr;
    var invPowBeta1:real(32) = (1.0 / (1.0 - beta1 ** opt.t)):real(32);
    var invPowBeta2:real(32) = (1.0 / (1.0 - beta2 ** opt.t)):real(32);

    for (ip,ig) in zip(D,opt.dom) {
        opt.accM[ig] = opt.accM[ig] * beta1 + opt.gradient[ig] * (1.0 - beta1);
        opt.accV[ig] = opt.accV[ig] * beta2 + opt.gradient[ig] ** 2 * (1.0 - beta2);
        var mHat: real(32) = opt.accM[ig] * invPowBeta1;
        var vHat: real(32) = opt.accV[ig] * invPowBeta2;
        parameter[ip] -= learningRate * mHat / (sqrt(vHat) + eps);
    }
    Set(0, D.size, opt.gradient, 0.0);
    opt.t += 1;
}

proc ComputeCrossEntropy(in start: int, in count: int, ref logits: [] real(32), ref target_token: int, ref grad: [] real(32)) {
    var max_logits = -inf;
    var buffer: [0..#tgtVocab] real(32);
    for i in 0..#count {
        max_logits = max(logits[start + i], max_logits);
    }

    var sum_exp = 0.0;
    for i in 0..#count {
        buffer[i] = exp(logits[start + i] - max_logits):real(32);
        sum_exp += buffer[i];
    }

    var loss = 0.0;
    for i in 0..#count {
        grad[start + i] = (buffer[i] / sum_exp) : real(32);
    }
    grad[start + i] -= 1;
    loss = -log2(buffer[i] / sum_exp);

    return loss;
}

proc _CrossEntropy(in start: int, in startToken: int, ref logits: [] real(32), ref targetToken: [] int, in tgtSeq: int, ref grad: [] real(32)) {
    var loss = 0.0;
    for i in 0..#tgtSeq {
        loss += ComputeCrossEntropy(
            start + i * tgtVocab,
            tgtVocab,
            logits, 
            targetToken[startToken + i], 
            grad);
    }
    return loss / tgtSeq;
}

proc CrossEntropy(ref logits: [] real(32), ref targetToken: [] int, ref tgtSeq: [] int, ref grad: [] real(32)) {
    var loss = 0.0;
    for i in 0..#batch {
        loss += _CrossEntropy(
            sequenceLength * tgtVocab * i,
            i * sequenceLength,
            logits,
            targetToken,
            tgtSeq[i],
            grad);
    }
    return loss / batch;
}