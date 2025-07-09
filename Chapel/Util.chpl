use Config;
use Math;

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
    opt.gradient = 0.0;
    opt.t += 1;
}