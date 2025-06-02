#ifndef CONFIG
#define CONFIG

static constexpr int warmupStep = 4000;
static constexpr float beta1 = 0.9;
static constexpr float beta2 = 0.98;
static constexpr float eps = 1e-9;

static constexpr int dModel = 24; // 512
static constexpr int head = 1; // 8
static constexpr int sequenceLength = 16;
static constexpr int qshape = 16;
static constexpr int dFF = 128; // 256

static constexpr int epoch = 2;
static constexpr int batch = 8;

#endif