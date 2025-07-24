config var lr: real(32) = 1e-4;
config var warmupStep: int = 4000;
config var beta1: real(32) = 0.9;
config var beta2: real(32) = 0.98;
config var eps: real(32) = 1e-9;

config var dModel: int = 32; // 512
config var head: int = 8; // 8
config var sequenceLength: int = 35;
config var dFF: int = 256; // 256
config var dropoutRate: real(32) = 0.1;
config var N: int = 6;
config var srcVocab: int = 128; // 15700
config var tgtVocab: int = 128; // 22470

config var epoch: int = 2;
config var batch: int = 8;

config param BLOCK_SIZE = 64;
config param trainingIteration = 25;
config param testingIteration = 10;


config param paramFileName = "../C++/Param/Transformer.param";