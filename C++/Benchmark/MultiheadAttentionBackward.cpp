#include <benchmark/benchmark.h>
#include "../Header.h"
#include "../Transformer.h"

static void escape(void *p) {
    asm volatile("" : : "g"(p) : "memory");
}

static void Bench(benchmark::State& state) {
    Tensor inputQ(batch * sequenceLength, dModel);
    Tensor inputK(batch * sequenceLength, dModel);
    Tensor inputV(batch * sequenceLength, dModel);
    Tensor output(batch * sequenceLength, dModel);
    Tensor outputGradient(batch * sequenceLength, dModel);
    Tensor inputGradientQ(batch * sequenceLength, dModel);
    Tensor inputGradientK(batch * sequenceLength, dModel);
    Tensor inputGradientV(batch * sequenceLength, dModel);
    int seq[8] = {64,64,64,64,64,64,64,64};

    MultiheadAttention layer;
    for(auto _ : state) {
        layer.backward(outputGradient, inputGradientQ, inputGradientK, inputGradientV, inputQ, inputK, inputV, output, LOOK_AHEAD, seq);
        escape(&output.data[0]);
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(Bench);

BENCHMARK_MAIN();