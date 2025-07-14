use Softmax;
use Config;
use Util;
use Matrix;
use Time;

param iterationTest = 200;
var t:stopwatch;


var input: [0..#(batch * head * sequenceLength * sequenceLength)] real(32);
var output: [0..#(batch * head * sequenceLength * sequenceLength)] real(32);
UniformInit(input, 1);
var softmax = new Softmax(batch * head* sequenceLength, sequenceLength);

t.start();
for i in 0..#iterationTest {
    softmax.forward(input, output);
}
t.stop();


writeln("Time per iteration : ", t.elapsed() / iterationTest);
writeln("Iteration per second : ", iterationTest / t.elapsed());