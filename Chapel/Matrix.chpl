use Time;
use Random;
use Config;
use Math;
use CTypes;
require "NpLoader.o", "NpLoader.h";

extern proc load_npz_flat(file: [] c_char, key:[] c_char, ref out_len: c_int, output :[] real(32)) : void;
proc loadNpzFlat(file: string, key: string, output: [?D] real(32)): void {
    var cLen: c_int;
    const lenPtr = c_ptrTo(cLen);

    const file_cstr = file.c_str();
    const key_cstr  = key.c_str();

    load_npz_flat(file, key, lenPtr, output);
}

var arr: [1..#dModel] real(32);
loadNpzFlat("../python/Testcase/LayerNorm/layer_norm_param.npz", "layerNorm.input", arr);
writeln(arr);





proc randMatrix(A: [?D] real(32)) {
    fillRandom(A, -1.0, 1.0);
}

var rng = new randomStream(eltType=real);

proc sampleNormal (in mu: real, in sigma: real) : real {
  const u1 = rng.next();
  const u2 = rng.next();
  const z0 = sqrt(-2 * log(u1)) * cos(2 * pi * u2);
  return mu + sigma * z0;
}

proc XavierUniformInit(ref parameter: [?D] real) : void {
    var size = D.size;
    var limit = sqrt(6.0 / size);
    fillRandom(parameter, -limit, limit);
}

proc UniformInit(ref parameter: [?D] real, in limit: real) : void {
    fillRandom(parameter, -limit, limit);
}

proc HeNormalInit(ref parameter: [?D] real) : void {
    var inD = parameter.shape[0];
    var stddev = sqrt(2.0 / inD);
    for p in parameter {
        p = sampleNormal(0.0, stddev);
    }
}

proc ApplyLookAheadMask(ref A:[?D] real(32), in seq: int, in x: real(32)) {
    for i in 0..#seq {
        for j in (i + 1)..<sequenceLength {
            A[i * sequenceLength + j] = x;
        }
    }
    for i in seq..<sequenceLength {
        for j in 0..#sequenceLength {
            A[i * sequenceLength + j] = x;
        }
    }
}

proc ApplyPaddingMask(ref A:[?D] real(32), in seq: int, in x: real(32)) {
    for i in 0..#seq {
        for j in seq..<sequenceLength {
            A[i * sequenceLength + j] = x;
        }
    }
    for i in seq..<sequenceLength {
        for j in 0..#sequenceLength {
            A[i * sequenceLength + j] = x;
        }
    }
}

proc ApplyCrossPaddingMask(ref A:[?D] real(32), in seq: int, in x: real(32)) {
    for i in 0..#sequenceLength {
        for j in seq..<sequenceLength {
            A[i * sequenceLength + j] = x;
        }
    }
}


proc Plus(ref A: [?D] real(32), ref B: [D] real(32), ref C:[D] real(32)) : void {
    for i in D {
        C[i] = A[i] + B[i];
    }
}

proc PlusProductInplace(ref A: [?D] real(32), ref B: [D] real(32), ref C: [D] real(32)) : void {
    for i in D {
        A[i] += B[i] * C[i];
    }
}

proc PlusReduce(ref A: [?D] real(32), out output: real(32)) : void {
    output = 0.0;
    for i in D {
        output += A[i];
    }
}

proc MaxReduce(ref A: [?D] real(32), out output) : void {
    output = -real(32).max;
    for i in D {
        output = max(A[i], output);
    }
}

proc ProductPlusReduce(ref A: [?D] real(32), ref B: [D] real(32), inout output: real(32)) : void {
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
        var x = A[i] - mean;
        output += x * x;
    }
    output /= D.size - 1;
    output = sqrt(output);
}

proc Mul(ref A: [?D] real(32), ref B: [D] real(32), ref C: [D] real(32)) : void {
    for i in D {
        C[i] = A[i] * B[i];
    }
}

proc Mul(ref A: [?D] real(32), in x: real(32), ref B:[D] real(32)) : void {
    for i in D {
        B[i] = A[i] * x;
    }
}

proc Div(ref A: [?D] real(32), in x: real(32), ref B:[D] real(32)) : void {
    Mul(A, 1.0 / x, B);
}





proc MatMulPlusAB(in d1: int, in d2: int, in d3: int,
    ref A:[?Da] real(32), ref B:[?Db] real(32), ref C:[?Dc] real(32)) : void {

    for ii in 0..<d1 by BLOCK_SIZE {
        for jj in 0..<d3 by BLOCK_SIZE {
            for kk in 0..<d2 by BLOCK_SIZE {
                
                var i = 0;
                while (i < BLOCK_SIZE && ii + i < d1) {
                    var k = 0;
                    while(k < BLOCK_SIZE && kk + k < d2) {
                        var j = 0;
                        while(j < BLOCK_SIZE && jj + j < d3) {
                            C[(ii + i) * d3 + (jj + j)] += A[(ii + i) * d2 + (kk + k)] * B[(kk + k) * d3 + (jj + j)];
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

    for ii in 0..<d1 by BLOCK_SIZE {
        for jj in 0..<d3 by BLOCK_SIZE {
            for kk in 0..<d2 by BLOCK_SIZE {
                
                var i = 0;
                while (i < BLOCK_SIZE && ii + i < d1) {
                    var k = 0;
                    while(k < BLOCK_SIZE && kk + k < d2) {
                        var j = 0;
                        while(j < BLOCK_SIZE && jj + j < d3) {
                            C[(ii + i) * d3 + (jj + j)] += A[(kk + k) * d1 + (ii + i)] * B[(kk + k) * d3 + (jj + j)];
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

    var BT : [Db] real(32);
    for ii in 0..<d2 by BLOCK_SIZE {
        for jj in 0..<d3 by BLOCK_SIZE {
            
            var i = 0;
            while(i < BLOCK_SIZE && ii + i < d2) {
                var j = 0;
                while(j < BLOCK_SIZE && jj + j < d3) {
                    BT[(ii + i) * d3 + jj + j] = B[(jj + j) * d2 + ii + i];
                    j+=1;
                }
                i +=1;
            }
        }
    }
    MatMulPlusAB(d1, d2, d3, A, BT, C);
}

