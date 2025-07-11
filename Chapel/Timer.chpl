use Timer;
use Sort;

config param timerSize = 10000;
var time: [0..#timerSize][0..#10000] real;
var it: [0..#timerSize] int;
var i : int;
var maxI : int;

var t: stopwatch;

proc RestartRecord() {
    t.start();
    i = 0;
}

proc CheckPoint() {
    t.top();
    time[i][it[i]] = t.elapsed() / 1000000;
    it[i] += 1;
    maxI = max(maxI, i);
    i += 1;
    t.start();
}

proc GetTime() {
    var res: [0..#maxI] real;
    for i in 0..#maxI {
        sort(time[i]);
        var start = min(it[i] * 0.1, 20): int;
        var end = max((it[i] * 9 + 9) / 10, it[i] - 20): int ;
        var mean: real;
        for j in start..<end {
            mean += time[i][j];
        }
        mean /= end - start;

        res[i] = mean;
    }
    return res;
}
