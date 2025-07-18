use LayerNorm;
use Config;
use Util;
use Matrix;
use Time;

param iterationTest = 40;
param stdTest = 10;
var t:stopwatch;



var tgtSeq: [0..#batch] int = 64;
var targetOutput: [0..#(batch * sequenceLength)] int = 100;
var output: [0..#(batch * sequenceLength * tgtVocab)] real(32);
var gradient: [0..#(batch * sequenceLength * tgtVocab)] real(32);

var time:[0..stdTest] real;
var totalTime: real;
var std: real;

for i in 0..#stdTest {
    t.reset();
    t.start();
    for j in 0..#iterationTest {
        CrossEntropy(output, targetOutput, tgtSeq, gradient);
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