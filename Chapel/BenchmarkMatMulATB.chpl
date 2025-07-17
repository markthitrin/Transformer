use LayerNorm;
use Config;
use Util;
use Matrix;
use Time;

param stdTest = 50;
var t:stopwatch;



proc testATB(in d1: int, in d2: int, in d3: int, in iterationTest: int = 100) {
    var A: [0..#(d2 * d1)] real(32);
    var B: [0..#(d2 * d3)] real(32);
    var C: [0..#(d1 * d3)] real(32);
    UniformInit(A, 1);
    UniformInit(B, 1);

    var time:[0..stdTest] real;
    var totalTime: real;
    var std: real;

    for i in 0..#stdTest {
        t.reset();
        t.start();
        for j in 0..#iterationTest {
            MatMulPlusATB(d1,d2,d3,A,B,C);
        }
        t.stop();
        time[i] = t.elapsed();
        totalTime += time[i];
    }
    var meanPerTest = totalTime / stdTest;
    for i in 0..#stdTest {
        var x = time[i] - meanPerTest;
        std += x * x;
    }
    std /= stdTest - 1;
    std = sqrt(std);


    writeln("mean Time per iteration : ", meanPerTest / iterationTest);
    writeln("std  Time per iteration : ", std / sqrt(iterationTest));
    writeln("mean Iteration per second : ", iterationTest / meanPerTest);
    writeln("std  Iteration per second : ",  iterationTest * iterationTest / (meanPerTest * meanPerTest) * (std / sqrt(iterationTest)));
}


// testATB(256,1024,32);
testATB(128,32,32);
testATB(128,4,128);
// testATB(32,1024,256);
// testATB(32,1024,22465);

