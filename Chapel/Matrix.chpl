use Config;
use IO;
use Math;
use Random;
use Time;
use Timer;

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

proc ApplyLookAheadMask(ref A:[?D] real(32), in seq: int, in x: real(32)) {
    ref Ar = A.reindex(0..#D.size);

    for i in 0..#seq {
        for j in (i + 1)..<sequenceLength {
            Ar[i * sequenceLength + j] = x;
        }
    }
    for i in seq..<sequenceLength {
        for j in 0..#sequenceLength {
            Ar[i * sequenceLength + j] = x;
        }
    }
}

proc ApplyPaddingMask(ref A:[?D] real(32), in seq: int, in x: real(32)) {
    ref Ar = A.reindex(0..#D.size);

    for i in 0..#seq {
        for j in seq..<sequenceLength {
            Ar[i * sequenceLength + j] = x;
        }
    }
    for i in seq..<sequenceLength {
        for j in 0..#sequenceLength {
            Ar[i * sequenceLength + j] = x;
        }
    }
}

proc ApplyCrossPaddingMask(ref A:[?D] real(32), in seq: int, in x: real(32)) {
    ref Ar = A.reindex(0..#D.size);
    
    for i in 0..#sequenceLength {
        for j in seq..<sequenceLength {
            Ar[i * sequenceLength + j] = x;
        }
    }
}

proc Copy(ref A: [?Da] real(32), ref B: [?Db] real(32)) {
    for (ia,ib) in zip(Da,Db) {
        B[ib] = A[ia];
    }
}

proc Set(ref A: [?Da] real(32), in x: real(32)) {
    for i in Da {
        A[i] = x;
    }
}

proc Plus(ref A: [?Da] real(32), ref B: [?Db] real(32), ref C:[?Dc] real(32)) : void {
    for (ia,ib,ic) in zip(Da,Db,Dc) {
        C[ic] = A[ia] + B[ib];
    }
}

proc PlusProductInplace(ref A: [?Da] real(32), ref B: [?Db] real(32), ref C: [?Dc] real(32)) : void {
    for (ia,ib,ic) in zip(Da,Db,Dc) {
        A[ia] += B[ib] * C[ic];
    }
}

proc PlusProductInplace(ref A: [?Da] real(32), ref B: [?Db] real(32), in C: real(32)) : void {
    for (ia,ib) in zip(Da,Db) {
        A[ia] += B[ib] * C;
    }
}

proc PlusReduce(ref A: [?D] real(32), out output: real(32)) : void {
    output = 0.0;
    for i in D {
        output += A[i];
    }
}

proc MaxReduce(ref A: [?D] real(32), out output: real(32)) : void {
    output = -inf;
    for i in D {
        output = max(A[i], output);
    }
}

proc ProductPlusReduce(ref A: [?D] real(32), ref B: [D] real(32), out output: real(32)) : void {
    output = 0.0;
    for i in D {
        output += A[i] * B[i];
    }
}

proc ExpPlusReduce(ref A: [?D] real(32), in maxValue, out output: real(32)) : void {
    output = 0.0;
    for i in D {
        output += exp(A[i] - maxValue);
    }
}

proc StdReduce(ref A: [?D] real(32), in mean: real(32), out output: real(32)) : void {
    output = 0.0;
    for i in D {
        var x: real(32) = A[i] - mean;
        output += x * x;
    }
    output /= D.size - 1;
    output = sqrt(output);
}

proc Mul(ref A: [?Da] real(32), ref B: [?Db] real(32), ref C: [?Dc] real(32)) : void {
    for (ia,ib,ic) in zip(Da,Db,Dc) {
        C[ic] = A[ia] * B[ib];
    }
}

proc Mul(ref A: [?Da] real(32), in x: real(32), ref B:[?Db] real(32)) : void {
    for (ia, ib) in zip(Da, Db) {
        B[ib] = A[ia] * x;
    }
}

proc Div(ref A: [?D] real(32), in x: real(32), ref B:[D] real(32)) : void {
    Mul(A, 1.0 / x, B);
}

proc MatMulPlusAB(in d1: int, in d2: int, in d3: int,
    ref A:[?Da] real(32), ref B:[?Db] real(32), ref C:[?Dc] real(32)) : void {
    
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
    ref A:[?Da] real(32), ref B:[?Db] real(32), ref C:[?Dc] real(32)) : void {

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
    ref A:[?Da] real(32), ref B:[?Db] real(32), ref C:[?Dc] real(32)) : void {

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
