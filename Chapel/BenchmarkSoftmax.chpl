use Softmax;
use Config;
use Util;
use Matrix;
use Time;

param iterationTest = 20;
param stdTest = 100;
var t:stopwatch;


var input: [0..#(batch * head * sequenceLength * sequenceLength)] real(32);
var output: [0..#(batch * head * sequenceLength * sequenceLength)] real(32);
UniformInit(input, 1);
var softmax = new Softmax(batch * head* sequenceLength, sequenceLength);

var time:[0..stdTest] real;
var totalTime: real;
var std: real;

for i in 0..#stdTest {
    t.reset();
    t.start();
    for j in 0..#iterationTest {
        softmax.forward(input, output);
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