config var lr: real(32) = 1e-4;
config var warmupStep: int = 4000;
config var beta1: real(32) = 0.9;
config var beta2: real(32) = 0.98;
config var eps: real(32) = 1e-9;

config var dModel: int = 512;
config var head: int = 8;
config var dFF: int = 2048;
config var dropoutRate: real(32) = 0.1;
config var N: int = 6;

config var batch: int = 8;
config var srcVocab: int = 15700;
config var tgtVocab: int = 22470;
config var sequenceLength: int = 256;

config param trainingIteration = 40;
config param testingIteration = 1;

config param trainingRatio = 0.7;

config param BLOCK_SIZE = 64;

config param paramFileName = "../C++/Param/Transformer.param";