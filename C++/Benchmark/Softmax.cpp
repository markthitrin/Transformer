#include <benchmark/benchmark.h>
#include "../Header.h"
#include "../Transformer.h"

static void escape(void *p) {
    asm volatile("" : : "g"(p) : "memory");
}

static void Bench(benchmark::State& state) {
    Tensor input(batch * head * sequenceLength, sequenceLength);
    Tensor output(batch * head * sequenceLength, sequenceLength);
    Softmax softmax;
    for(auto _ : state) {
        softmax.forward(input, output);
        escape(&output.data[0]);
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(Bench);

BENCHMARK_MAIN();