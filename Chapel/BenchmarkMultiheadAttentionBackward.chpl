use MultiheadAttention;
use Config;
use Util;
use Matrix;
use Time;

param iterationTest = 400;
param stdTest = 10;
var t:stopwatch;





var inputQ: [0..#(batch * sequenceLength * dModel)] real(32);
var inputK: [0..#(batch * sequenceLength * dModel)] real(32);
var inputV: [0..#(batch * sequenceLength * dModel)] real(32);
var output: [0..#(batch * sequenceLength * dModel)] real(32);
var outputGradient: [0..#(batch * sequenceLength * dModel)] real(32);
var inputGradientQ: [0..#(batch * sequenceLength * dModel)] real(32);
var inputGradientK: [0..#(batch * sequenceLength * dModel)] real(32);
var inputGradientV: [0..#(batch * sequenceLength * dModel)] real(32);
var seq: [0..#8] int;
var layer = new MultiheadAttention();

var time:[0..stdTest] real;
var totalTime: real;
var std: real;

for i in 0..#stdTest {
    t.reset();
    t.start();
    for j in 0..#iterationTest {
        layer.backward(outputGradient, inputGradientQ, inputGradientK, inputGradientV, inputQ, inputK, inputV, output, MaskType.LOOK_AHEAD, seq);
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