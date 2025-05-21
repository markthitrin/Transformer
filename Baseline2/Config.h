#ifndef CONFIG
#define CONFIG

extern const int warmupStep;
extern const float b1;
extern const float b2;
extern const float eps;

extern const int dModel;
extern const int head;
extern const int sequenceLength;
extern const int qshape;
extern const int dFF;

extern const int epoch;
extern const int batch;

#define L1_SIZE = 256 * 1024
#define L2_SIZE = 4 * 1024 * 1024
#define L3_SIZE = 8 * 1024 * 1024

#endif