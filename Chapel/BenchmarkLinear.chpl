use Linear;
use Config;
use Util;
use Matrix;
use Time;

param iterationTest = 2000;
var t:stopwatch;





var input: [0..#(batch * sequenceLength * dFF)] real(32);
var output: [0..#(batch * sequenceLength * dFF)] real(32);
UniformInit(input, 1);
var linear = new Linear(dFF, dFF);

t.start();
for i in 0..#iterationTest {
    linear.forward(input, output);
}
t.stop();


writeln("Time per iteration : ", t.elapsed() / iterationTest);
writeln("Iteration per second : ", iterationTest / t.elapsed());