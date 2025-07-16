#include <benchmark/benchmark.h>
#include "../Header.h"
#include "../Transformer.h"

static void escape(void *p) {
    asm volatile("" : : "g"(p) : "memory");
}

static void Bench(benchmark::State& state) {
    Tensor input(batch * sequenceLength, dModel);
    Tensor output(batch * sequenceLength, dModel);
    DropOut dropout(batch * sequenceLength, dModel);
    for(auto _ : state) {
        dropout.forward(input, output);
        escape(&output.data[0]);
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(Bench);

BENCHMARK_MAIN();