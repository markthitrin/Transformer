use MemDiagnostics;
use Util;
use Math;
use Matrix;
use Config;
use CTypes;

var A: [0..1000000] real(32);
var B: [0..1000000] real(32);
var x: real(32);
for j in 0..50000 do 
PlusReduce3(0,5000,A,x);
   
writeln(x);


