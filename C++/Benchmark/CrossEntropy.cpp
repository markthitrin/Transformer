#include <benchmark/benchmark.h>
#include "../Header.h"
#include "../Transformer.h"

static void escape(void *p) {
    asm volatile("" : : "g"(p) : "memory");
}

static void Bench(benchmark::State& state) {
    Tensor logits(batch * sequenceLength, tgtVocab);
    Tensor gradient(batch * sequenceLength, tgtVocab);
    int targetOutput[batch * sequenceLength];
    for(int q = 0 ;q < batch * sequenceLength;q++) {
        targetOutput[q] = 100; 
    }
    int tgtSeq[batch];
    for(int q = 0;q < batch;q++) {
        tgtSeq[q] = 64;
    }
    for(auto _ : state) {
        CrossEntropy(logits, targetOutput, tgtSeq, gradient);
        escape(&gradient.data[0]);
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(Bench);

BENCHMARK_MAIN();