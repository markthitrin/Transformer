use Sort;
use Time;

config param timerSize = 10000;
var time: [0..#timerSize][-1..#10000] real;
var round: int = -1;
var i : int;
var maxI : int;

var t: stopwatch;

proc RestartRecord() {
    t.stop();
    i = 0;
    round += 1;
    t.reset();
    t.start();
}

proc CheckPoint() {
    t.stop();
    time[i][round] = t.elapsed() * 1000000;
    i += 1;
    maxI = max(maxI, i);
    t.reset();
    t.start();
}

proc GetTime() {
    var res: [0..#maxI] real;
    for i in 0..#maxI {
        sort(time[i][0..#round]);
        var start = min(round * 0.1, 20): int;
        var end = max((round * 9 + 9) / 10, round - 20): int ;
        var mean: real;
        for j in start..<end {
            mean += time[i][j];
        }
        mean /= end - start;

        res[i] = mean;
    }
    return res;
}

proc GetTimeStd() {
    var res: [0..#maxI] real;
    for i in 0..#maxI {
        sort(time[i][0..#round]);
        var start = min(round * 0.1, 20): int;
        var end = max((round * 9 + 9) / 10, round - 20): int ;
        var mean: real;
        var std: real;
        for j in start..<end {
            mean += time[i][j];
        }
        mean /= end - start;

        for j in start..<end {
            var x = (time[i][j] - mean);
            std += x * x;
        }
        std /= (end - start) - 1;
        std = sqrt(std);

        res[i] = std;
    }
    return res;
}