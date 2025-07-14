#include <benchmark/benchmark.h>
#include "../Header.h"
#include "../Transformer.h"

static void escape(void *p) {
    asm volatile("" : : "g"(p) : "memory");
}

static void Bench(benchmark::State& state) {
    Tensor input(batch * sequenceLength, dFF);
    Tensor output(batch * sequenceLength, dFF);
    Linear linear(dFF, dFF);
    for(auto _ : state) {
        linear.forward(input, output);
        escape(&output.data[0]);
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(Bench);

BENCHMARK_MAIN();