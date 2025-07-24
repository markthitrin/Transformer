use MemDiagnostics;
use Util;
use Math;
use Matrix;
use Config;
use CTypes;
use Random;

var A: [0..#1000000] int;
var dom = {0..5};
var rng: [dom] randomStream(int);
dom = 0..6;
var block = A.domain.size / 4;
coforall i in 0..#4 {
    rng[i].fill(A[(i * block)..#block], 0, 50);
}
writeln(A[0..10]);


