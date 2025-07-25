#include <benchmark/benchmark.h>
#include "../Header.h"
#include "../Transformer.h"

static void escape(void *p) {
    asm volatile("" : : "g"(p) : "memory");
}

static void Bench(benchmark::State& state) {
    Tensor A(batch * sequenceLength, dModel);
    Tensor B(batch * sequenceLength, dModel);
    Tensor C(batch * sequenceLength, dModel);
    for(auto _ : state) {
       #pragma omp parallel for schedule(static)
        for(int i = 0;i < C.row * C.col;i++) {
            C[i] = A[i] * B[i];
        }
    }
    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(Bench);

BENCHMARK_MAIN();