use MemDiagnostics;
use Util;
use Math;
use Matrix;
use Config;
use CTypes;

var A: [0..10000000] real(32);
var B = A;
B = exp(A);
