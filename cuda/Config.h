#ifndef CONFIG
#define CONFIG

static constexpr float lr = 1e-4;
static constexpr std::size_t warmupStep = 4000;
static constexpr float beta1 = 0.9;
static constexpr float beta2 = 0.999;
static constexpr float eps = 1e-9;

static constexpr std::size_t dModel = 32; // 512
static constexpr std::size_t head = 8; // 8
static constexpr std::size_t sequenceLength = 35;
static constexpr std::size_t dFF = 256; // 256
static constexpr float dropoutRate = 0.1;
static constexpr std::size_t N = 6;
static constexpr std::size_t srcVocab = 6;
static constexpr std::size_t tgtVocab = 6;

static constexpr std::size_t epoch = 2;
static constexpr std::size_t batch = 8;



static constexpr std::size_t MATMUL_BLOCKSIZE = 16;

#endif