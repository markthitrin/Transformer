config param lr: real(32) = 1e-4;
config param warmupStep: int = 4000;
config param beta1: real(32) = 0.9;
config param beta2: real(32) = 0.98;
config param eps: real(32) = 1e-9;

config param dModel: int = 32; // 512
config param head: int = 8; // 8
config param sequenceLength: int = 128;
config param dFF: int = 256; // 256
config param dropoutRate: real(32) = 0.1;
config param N: int = 6;
config param srcVocab: int = 15700; // 15700
config param tgtVocab: int = 22470; // 22470

config param epoch: int = 2;
config param batch: int = 8;

config param BLOCK_SIZE = 64;
config param trainingIteration = 600;
config param testingIteration = 10;


config param paramFileName = "../C++/Param/Transformer.param";