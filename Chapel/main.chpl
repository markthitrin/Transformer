use Data;
use Transformer;
use Config;
use Util;
use IO;
use Timer;

proc readDataFile(ref dataDomain: domain(2), ref src: [dataDomain] int, ref tgt: [dataDomain] int) {
    var file = open("opus_books_tokenized.txt", ioMode.r);
    var fileReader = file.reader();
    var i = 0;
    while !fileReader.atEOF() {
        if(dataDomain.dim(0).high < i) {
            var h = src.dim(0).size * 2;
            dataDomain = {0..h, dataDomain.dim(1)};
        }
        fileReader.read(src[i, -1]);
        for j in 0..#src[i,-1] {
            fileReader.read(src[i,j]);
        }
        fileReader.read(tgt[i, -1]);
        for j in 0..#tgt[i,-1] {
            fileReader.read(tgt[i,j]);
        }
        i += 1;
    }
    dataDomain = {0..(i - 1), dataDomain.dim(1)};

    fileReader.close();
    file.close();
}

proc readTranslateFile(ref transDomain: domain(1) ,ref translator: [?Dt] string, lang: string) {
    var file = open("Translate_" + lang + ".txt", ioMode.r);
    var fileReader = file.reader();
    var i = 0;
    while !fileReader.atEOF() {
        if(transDomain.high < i) {
            var h = transDomain.size * 2;
            transDomain = {0..h};
        }
        fileReader.read(translator[i]);
        i += 1;
    }
    transDomain = {0..(i - 1)};

    fileReader.close();
    file.close();
}

proc printSentence(ref translator: [?Dt] string, ref token: [?D] int, seq: int) {
    for i in D {
        if i - D.low == seq {
            break;
        }
        write(translator[token[i]]," ");
    }
    writeln();
    writeln();
}

proc split(ref dataDomain: domain(2), ref src: [dataDomain] int, ref tgt: [dataDomain] int, 
    ref trainDomain: domain(2), ref srcTrain: [trainDomain] int, ref tgtTrain: [trainDomain] int,
    ref testDomain: domain(2), ref srcTest: [testDomain] int, ref tgtTest: [testDomain] int,
    trainRatio: real = 0.7) {
    
    var cut = (dataDomain.dim(0).size * trainRatio):int;
    var testSize = dataDomain.dim(0).size - cut;

    trainDomain = {0..#cut, testDomain.dim(1)};
    testDomain = {0..#testSize, testDomain.dim(1)};
    srcTrain = src[0..#cut, ..];
    tgtTrain = tgt[0..#cut, ..];
    srcTest = src[cut..#testSize, ..];
    tgtTest = tgt[cut..#testSize, ..];
}

proc getOutputToken(ref output: [?D] real(32)) {
    ref outputr = output.reindex(0..#output.domain.size);
    var tokens: [0..#sequenceLength] int;
    for i in 0..#sequenceLength {
        var maxValue = 0.0;
        var maxPos = 0;
        for j in 0..#tgtVocab {
            if(maxValue < outputr[i * tgtVocab + j]) {
                maxValue = outputr[i * tgtVocab + j];
                maxPos = j;
            }
        }
        tokens[i] = maxPos;
    }
    return tokens;
}

proc main() {
    var dataDomain: domain(2) = {0..1, -1..#1000};
    var trainDomain: domain(2)= {0..1, -1..#1000};
    var testDomain: domain(2)= {0..1, -1..#1000};
    var enTransDomain: domain(1)= {0..1};
    var itTransDomain: domain(1)= {0..1};
    var src: [dataDomain] int;
    var tgt: [dataDomain] int;
    var srcTrain: [trainDomain] int;
    var tgtTrain: [trainDomain] int;
    var srcTest: [testDomain] int;
    var tgtTest: [testDomain] int;
    var translatorEn: [enTransDomain] string;
    var translatorIt: [itTransDomain] string;
    writeln("read data");
    readDataFile(dataDomain, src, tgt);
    readTranslateFile(enTransDomain, translatorEn, "en");
    readTranslateFile(itTransDomain, translatorIt, "it");
    writeln("split data");
    split(dataDomain, src, tgt, trainDomain, srcTrain, tgtTrain, testDomain, srcTest, tgtTest);
    writeln("Create the model");
    var model = new Transformer();

    {   // Training Section
        var encoderInput: [0..#(batch * sequenceLength)] int;
        var srcSeq: [0..#batch] int;
        var decoderInput: [0..#(batch * sequenceLength)] int;
        var tgtSeq: [0..#batch] int;
        var targetOutput: [0..#(batch * sequenceLength)] int;
        var output: [0..#(batch * sequenceLength * tgtVocab)] real(32);
        var gradient: [0..#(batch * sequenceLength * tgtVocab)] real(32);
        writeln("Start Training");
        for i in 0..#trainingIteration {
            getData(srcTrain, tgtTrain, encoderInput, decoderInput, targetOutput, srcSeq, tgtSeq);
            RestartRecord();
            model.forward(encoderInput, decoderInput, output, srcSeq, tgtSeq);
            var loss = CrossEntropy(output, targetOutput, tgtSeq, gradient);
            CheckPoint();
            model.backward(gradient, encoderInput, decoderInput, srcSeq, tgtSeq);     
            model.updateParameter();
            CheckPoint();
            writeln("Iteration [", i, " / ", trainingIteration, "] loss : ", loss);
        }
    }

    writeln("Time Recorded ====================================\n\n");
    var rec = GetTime();
    for i in rec.domain {
        writeln(rec[i]);
    }

    writeln("Time Recorded std ================================\n\n");
    var recstd = GetTimeStd();
    for i in recstd.domain {
        writeln(recstd[i]);
    }

    {   // Evaluation Section 
        var encoderInput: [0..#(batch * sequenceLength)] int;
        var srcSeq: [0..#batch] int;
        var decoderInput: [0..#(batch * sequenceLength)] int;
        var tgtSeq: [0..#batch] int;
        var targetOutput: [0..#(batch * sequenceLength)] int;
        var output: [0..#(batch * sequenceLength * tgtVocab)] real(32);
        writeln("Start evaluation");
        for i in 0..#testingIteration {
            getData(srcTest, tgtTest, encoderInput, decoderInput, targetOutput, srcSeq, tgtSeq);
            model.predict(encoderInput, decoderInput, output, srcSeq, tgtSeq);
            for j in 0..#batch {
                var outputTokens =  getOutputToken(output[(j * sequenceLength * tgtVocab)..#(sequenceLength * tgtVocab)]);
                writeln("English ::::::::::::::::::::::::::::::::::::::::::::::::::::\n"); 
                printSentence(translatorEn, encoderInput[(j * sequenceLength)..#sequenceLength], srcSeq[j]);
                writeln("Italian (target) :::::::::::::::::::::::::::::::::::::::::::\n");
                printSentence(translatorIt, targetOutput[(j * sequenceLength)..#sequenceLength], tgtSeq[j]);
                writeln("Italian (predicted) ::::::::::::::::::::::::::::::::::::::::\n");
                printSentence(translatorIt, outputTokens, tgtSeq[j]);
                writeln("\n\n\n");
            }
        }
    }
}

/*
$CHPL_HOME/modules/standard/Random.chpl:879: internal error: RES-VIS-ONS-0424 chpl version 2.4.0
Note: This source location is a guess.

Internal errors indicate a bug in the Chapel compiler,
and we're sorry for the hassle.  We would appreciate your reporting this bug --
please see https://chapel-lang.org/bugs.html for instructions.  In the meantime,
the filename + line number above may be useful in working around the issue.
*/