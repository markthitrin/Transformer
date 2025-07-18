#include <benchmark/benchmark.h>
#include "../Header.h"
#include "../Transformer.h"

static void escape(void *p) {
    asm volatile("" : : "g"(p) : "memory");
}

static void Bench(benchmark::State& state) {
    Tensor input(batch * sequenceLength, dFF);
    Tensor outputGradient(batch * sequenceLength, dFF);
    Tensor inputGradient(batch * sequenceLength, dFF);
    ReLU relu;
    for(auto _ : state) {
        relu.backward(outputGradient, outputGradient, input);
        escape(&input.data[0]);
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(Bench);

BENCHMARK_MAIN();