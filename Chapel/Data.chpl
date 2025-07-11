use Config;
use Random;

var rng = new randomStream(int);
proc randomInt(low: int, high: int): int {
    return rng.next(low, high);
}

proc getData(
    ref srcTokens: [?Ds] int, ref tgtToken: [?Dt] int,
    ref inpute: [?Di] int, ref inputd: [Di] int, ref target: [Di] int,
    ref srcSeq: [?Dse] int, ref tgtSeq: [Dse] int) {

    var dataSize = srcTokens.dim(0).size;
    var i = 0;
    while i < batch {
        var pos = randomInt(0,dataSize);
        var srclen = srcTokens[pos, -1];
        var tgtlen = randomInt(0, tgtToken[pos, -1]);
        if(srclen > sequenceLength - 1 || tgtlen + 1 > sequenceLength) {
            i -= 1;
            continue;
        }

        for j in 0..<srclen {
            inpute[i * sequenceLength + j] = srcTokens[pos,j];
        }
        for j in srclen..<sequenceLength {
            inpute[i * sequenceLength + j] =  1;
        }

        inputd[i * sequenceLength] = 2;
        for j in 1..<(tgtlen + 1) {
            inputd[i * sequenceLength + j] = tgtToken[pos, j - 1];
        }
        for j in (tgtlen + 1)..<sequenceLength {
            inputd[i * sequenceLength + j] = 1;
        }

        for j in 0..<tgtlen {
            target[i * sequenceLength + j] = tgtToken[pos,j];
        }
        if tgtToken[pos,-1] == tgtlen {
            target[i * sequenceLength + tgtlen] = 3;
        }
        else {
            target[i * sequenceLength + tgtlen] = tgtToken[pos, tgtlen];
        }
        for j in (tgtlen + 1)..<sequenceLength {
            target[i * sequenceLength + j] = 1;
        }

        srcSeq[i] = srclen;
        tgtSeq[i] = tgtlen + 1;

        i+=1;
    }
}
