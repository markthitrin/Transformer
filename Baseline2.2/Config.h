#ifndef CONFIG
#define CONFIG

static constexpr float lr = 1e-4;
static constexpr int warmupStep = 4000;
static constexpr float beta1 = 0.9;
static constexpr float beta2 = 0.999;
static constexpr float eps = 1e-9;

static constexpr int dModel = 24; // 512
static constexpr int head = 3; // 8
static constexpr int sequenceLength = 32;
static constexpr int dFF = 96; // 256
static constexpr float dropoutRate = 0.1;
static constexpr int N = 4;
static constexpr int srcVocab = 15699;
static constexpr int tgtVocab = 22465;

static constexpr int epoch = 2;
static constexpr int batch = 8;

static constexpr float trainingRatio = 0.7;

#endif