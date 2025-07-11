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
    Set(opt.gradient, 0.0);
    opt.t += 1;
}

proc ComputeCrossEntropy(ref logits: [?D] real(32), ref target_token: int, ref grad: [D] real(32)) {
    var max_logits = -inf;
    for i in D {
        max_logits = max(logits[i], max_logits);
    }

    var sum_exp = 0.0;
    for i in D {
        sum_exp += exp(logits[i] - max_logits);
    }

    var loss = 0.0;
    for i in D {
        var prob = (exp(logits[i] - max_logits) / sum_exp) : real(32);
        grad[i] = prob;
        if(i - D.low == target_token) {
            loss = -log2(prob + 1e-9);
            grad[i] -= 1;
        }
    }

    return loss;
}

proc _CrossEntropy(ref logits: [?D] real(32), ref targetToken: [?Dt] int, in tgtSeq: int, ref grad: [D] real(32)) {
    var loss = 0.0;
    var ground = D.low;
    ref targetTokenr = targetToken.reindex(0..#sequenceLength);
    for i in 0..#tgtSeq {
        loss += ComputeCrossEntropy(
            logits[(i * tgtVocab + ground)..#tgtVocab], 
            targetTokenr[i], 
            grad[(i * tgtVocab + ground)..#tgtVocab]);
    }
    return loss / tgtSeq;
}

proc CrossEntropy(ref logits: [?D] real(32), ref targetToken: [?Dt] int, ref tgtSeq: [?Ds] int, ref grad: [D] real(32)) {
    var loss = 0.0;
    var block = (sequenceLength * tgtVocab);
    for i in 0..#batch {
        loss += _CrossEntropy(
            logits[(i * block)..#block],
            targetToken[(i * sequenceLength)..#sequenceLength],
            tgtSeq[i],
            grad[(i * block)..#block]);
    }
    return loss / batch;
}