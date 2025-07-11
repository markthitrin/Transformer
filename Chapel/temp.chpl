use MemDiagnostics;
use Util;
use Math;
use Matrix;
use Config;
use CTypes;

var A: [0..1100000] real(32);
var B = A;
for i in 0..1000000 {
    B = A;
}
