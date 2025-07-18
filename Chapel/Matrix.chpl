use Config;
use IO;
use Math;
use Random;
use Time;

var file = open(paramFileName, ioMode.r);
var fileReader = file.reader();
proc loadM(ref A:[?D] real(32)) {
    for i in D {
        var x: real(32);
        fileReader.read(x);
        A[i] = x;
    }
}
proc loadM(ref A:[?D] int) {
    for i in D {
        var x: int;
        fileReader.read(x);
        A[i] = x;
    }
}


var rng = new randomStream(eltType=real(32));

proc sampleNormal (in mu: real(32), in sigma: real(32)) : real(32) {
  const u1:real(32) = rng.next();
  const u2:real(32) = rng.next();
  const z0:real(32) = (sqrt(-2 * log(u1)) * cos(2 * pi * u2)):real(32);
  return mu + sigma * z0;
}

proc XavierUniformInit(ref parameter: [?D] real(32)) : void {
    var size = D.size;
    var limit = sqrt(6.0 / size):real(32);
    fillRandom(parameter, -limit, limit);
}

proc UniformInit(ref parameter: [?D] real(32), in limit: real(32)) : void {
    fillRandom(parameter, -limit, limit);
}

proc HeNormalInit(ref parameter: [?D] real(32)) : void {
    var inD = parameter.shape[0];
    var stddev:real(32) = sqrt(2.0 / inD):real(32);
    for p in parameter {
        p = sampleNormal(0.0:real(32), stddev);
    }
}

proc ApplyLookAheadMask(in start: int, ref A:[] real(32), in seq: int, in x: real(32)) {

    for i in 0..#seq {
        for j in (i + 1)..<sequenceLength {
            A[start + i * sequenceLength + j] = x;
        }
    }
    for i in seq..<sequenceLength {
        for j in 0..#sequenceLength {
            A[start + i * sequenceLength + j] = x;
        }
    }
}

proc ApplyPaddingMask(in start: int, ref A:[] real(32), in seq: int, in x: real(32)) {

    for i in 0..#seq {
        for j in seq..<sequenceLength {
            A[start + i * sequenceLength + j] = x;
        }
    }
    for i in seq..<sequenceLength {
        for j in 0..#sequenceLength {
            A[start + i * sequenceLength + j] = x;
        }
    }
}

proc ApplyCrossPaddingMask(in start: int, ref A:[] real(32), in seq: int, in x: real(32)) {
    
    for i in 0..#sequenceLength {
        for j in seq..<sequenceLength {
            A[start + i * sequenceLength + j] = x;
        }
    }
}

proc Copy(in starta: int, in startb: int, in count: int, ref A: [] real(32), ref B: [] real(32)) {
    
    for i in 0..#count {
        B[startb + i] = A[starta + i];
    }
}

proc Set(in starta: int, in count:int, ref A: [] real(32), in x: real(32)) {
    for i in 0..#count {
        A[starta + i] = x;
    }
}

proc Plus(in starta: int, in startb: int, in startc: int, in count: int,
    ref A: [] real(32), ref B: [] real(32), ref C:[] real(32)) : void {
    for i in 0..#count {
        C[startc + i] = A[starta + i] + B[startb + i];
    }
}

proc PlusProductInplace(in starta: int, in startb: int, in startc: int, in count: int,
    ref A: [] real(32), ref B: [] real(32), ref C: [] real(32)) : void {
    for i in 0..#count {
        A[starta + i] += B[startb + i] * C[startc + i];
    }
}

proc PlusProductInplace(in starta: int, in startb: int, in count: int,
    ref A: [] real(32), ref B: [] real(32), in C: real(32)) : void {
    for i in 0..#count {
        A[starta + i] += B[startb + i] * C;
    }
}

proc PlusReduce(in starta: int, in count: int,
    ref A: [] real(32), out output: real(32)) : void {
    output = 0.0;
    for i in 0..#count {
        output += A[starta + i];
    }
}

proc MaxReduce(in starta: int, in count: int,
    ref A: [] real(32), out output: real(32)) : void {
    output = -inf;
    for i in 0..#count {
        output = max(A[starta + i], output);
    }
}

proc ProductPlusReduce(in starta: int, in startb: int, in count: int,
    ref A: [] real(32), ref B: [] real(32), out output: real(32)) : void {
    output = 0.0;
    for i in 0..#count {
        output += A[starta + i] * B[startb + i];
    }
}

proc ExpPlusReduce(in starta: int, in count: int,
    ref A: [] real(32), in maxValue, out output: real(32)) : void {
    output = 0.0;
    for i in 0..#count {
        output += exp(A[starta + i] - maxValue);
    }
}

proc StdReduce(in starta: int, in count: int,
    ref A: [] real(32), in mean: real(32), out output: real(32)) : void {
    output = 0.0;
    for i in 0..#count {
        var x: real(32) = A[starta + i] - mean;
        output += x * x;
    }
    output /= count - 1;
    output = sqrt(output);
}

proc Mul(in starta: int, in startb, in count: int,
    ref A: [] real(32), in x: real(32), ref B:[] real(32)) : void {
    for i in 0..#count {
        B[startb + i] = A[starta + i] * x;
    }
}

proc Mul(in starta: int, in startb: int, in count: int,
    ref A: [] real(32), in x: real(32), ref B:[] real(32)) : void {
    for i in 0..#count {
        B[startb + i] = A[starta + i] * x;
    }
}

proc Div(in starta: int, in startb: int, in count:int, ref A: [?D] real(32), in x: real(32), ref B:[D] real(32)) : void {
    Mul(starta, startb, count, A, 1.0 / x, B);
}

proc Exp(in starta: int, in startb: int, in count: int,
    ref A: [] real(32), in maxValue: real(32), ref B:[] real(32)) : void {
    for i in 0..#count {
        B[startb + i] = exp(A[starta + i] - maxValue);
    }
}

proc MatMulPlusAB(in d1: int, in d2: int, in d3: int,
    const ref A:[] real(32), const ref B:[] real(32), ref C:[] real(32)) : void {
    
    ref Ar = A.reindex(0..#(d1 * d2));
    ref Br = B.reindex(0..#(d2 * d3));
    ref Cr = C.reindex(0..#(d1 * d3));

    for ii in 0..<d1 by BLOCK_SIZE {
        for jj in 0..<d3 by BLOCK_SIZE {
            for kk in 0..<d2 by BLOCK_SIZE {
                
                var i = 0;
                while (i < BLOCK_SIZE && ii + i < d1) {
                    var k = 0;
                    while(k < BLOCK_SIZE && kk + k < d2) {
                        var j = 0;
                        while(j < BLOCK_SIZE && jj + j < d3) {
                            Cr[(ii + i) * d3 + (jj + j)] += Ar[(ii + i) * d2 + (kk + k)] * Br[(kk + k) * d3 + (jj + j)];
                            j += 1;
                        }
                        k += 1;
                    }
                    i += 1;
                }
            }
        }
    } 
}

proc MatMulPlusATB(in d1: int, in d2: int, in d3: int,
    ref A:[] real(32), ref B:[] real(32), ref C:[] real(32)) : void {

    ref Ar = A.reindex(0..#(d1 * d2));
    ref Br = B.reindex(0..#(d2 * d3));
    ref Cr = C.reindex(0..#(d1 * d3));

    for ii in 0..<d1 by BLOCK_SIZE {
        for jj in 0..<d3 by BLOCK_SIZE {
            for kk in 0..<d2 by BLOCK_SIZE {
                
                var i = 0;
                while ((i < BLOCK_SIZE) & (ii + i < d1)) {
                    var k = 0;
                    while((k < BLOCK_SIZE) & (kk + k < d2)) {
                        var j = 0;
                        while((j < BLOCK_SIZE) & (jj + j < d3)) {
                            Cr[(ii + i) * d3 + (jj + j)] += Ar[(kk + k) * d1 + (ii + i)] * Br[(kk + k) * d3 + (jj + j)];
                            j += 1;
                        }
                        k += 1;
                    }
                    i += 1;
                }
            }
        }
    } 
}

proc MatMulPlusABT(in d1: int, in d2: int, in d3: int,
    ref A:[] real(32), ref B:[] real(32), ref C:[] real(32)) : void {

    ref Ar = A.reindex(0..#(d1 * d2));
    ref Br = B.reindex(0..#(d2 * d3));
    ref Cr = C.reindex(0..#(d1 * d3));

    var BT : [0..#(d2 * d3)] real(32);
    for ii in 0..<d2 by BLOCK_SIZE {
        for jj in 0..<d3 by BLOCK_SIZE {
            
            var i = 0;
            while(i < BLOCK_SIZE && ii + i < d2) {
                var j = 0;
                while(j < BLOCK_SIZE && jj + j < d3) {
                    BT[(ii + i) * d3 + jj + j] = Br[(jj + j) * d2 + ii + i];
                    j+=1;
                }
                i +=1;
            }
        }
    }
    MatMulPlusAB(d1, d2, d3, A, BT, C);
}

proc PrintTestResult(text: string, ref A: [?D] real(32), ref B:[D] real(32)) {
    var sum = 0.0;
    for i in D {
        sum += abs(A[i] - B[i]);
    }
    sum /= D.size;
    writeln("Test result [", text, "] : ", sum);

    var count = 0;
    for i in D do
        if abs(A[i] - B[i]) >= 0.00009 {
            writeln("\t\t", A[i], " :: ", B[i], "\t(", i, ")");
            count += 1;
            if count >= 6 then break;
        }

    writeln();
}
