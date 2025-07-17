#ifndef CONFIG
#define CONFIG

static constexpr float lr = 1e-4;
static constexpr int warmupStep = 4000;
static constexpr float beta1 = 0.9;
static constexpr float beta2 = 0.999;
static constexpr float eps = 1e-9;

static constexpr int dModel = 32; // 512
static constexpr int head = 8; // 8
static constexpr int sequenceLength = 35;
static constexpr int dFF = 256; // 256
static constexpr float dropoutRate = 0.1;
static constexpr int N = 6;
static constexpr int srcVocab = 128; // 15700
static constexpr int tgtVocab = 128; // 22470

static constexpr int epoch = 2;
static constexpr int trainingIteration = 200;
static constexpr int testingIteration = 10;
static constexpr int batch = 8;

static constexpr float trainingRatio = 0.7;

static constexpr int BLOCK_SIZE = 64;

static constexpr bool verbose = false;

#endif