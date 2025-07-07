use Math;

record AdamOptGradient {
    proc init(ref parameter: [?D] real(32)) {
        dom = D;
        t = 0;
    }
    
    var dom: domain(1);
    var gradient: [dom] real(32);
    var accM: [dom] real(32);
    var accV: [dom] real(32);
    var t: int;
};

proc AdamOpt(ref parameter: [?D] real(32), ref optimizer: AdamOptGradient1) : void {
    var learningRate: lr;
    var invPowBeta1 = 1.0 / (1.0 - beta1 ** optimizer.t);
    var invPowBeta2 = 1.0 / (1.0 - beta2 ** optimizer.t);

    optimizer.accM = optimizer.accM * beta1 + optimizer.gradient * (1.0 - beta1);
    optimizer.accV = optimizer.accV * beta2 + optimizer.gradient ** 2 * (1.0 - beta2);
    parameter -= learningRate * (optimizer.accM * invPowBeta1) / (sqrt(optimizer.accV * invPowBeta2) + eps);
    
    optimizer.gradient = 0.0;
    optimizer.t += 1;
}