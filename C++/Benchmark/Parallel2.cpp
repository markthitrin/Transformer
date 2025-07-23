#include <benchmark/benchmark.h>
#include <omp.h>
#include "../Header.h"
#include "../Transformer.h"

static void escape(void *p) {
    asm volatile("" : : "g"(p) : "memory");
}

static void Bench(benchmark::State& state) {
    const int d = state.range(0);
    int max_threads = d;
    for(auto _ : state) {
        #pragma omp parallel for num_threads(max_threads) schedule(static)
        for(int q = 0;q < max_threads;q++) {
            float c = 4;
            escape(&c);
        }
    }
    state.SetItemsProcessed(state.iterations());
}


static void CustomArgs(benchmark::internal::Benchmark* b) {
    b->Args({1});
    b->Args({2});
    b->Args({3});
    b->Args({4});
    b->Args({5});
    b->Args({6});
    b->Args({7});
    b->Args({8});
    b->Args({9});
    b->Args({10});
    b->Args({11});
    b->Args({12});
    b->Args({13});
    b->Args({14});
    b->Args({15});
    b->Args({16});
    b->Args({17});
    b->Args({18});
    b->Args({19});
    b->Args({20});
    b->Args({21});
    b->Args({22});
    b->Args({23});
    b->Args({24});
    b->Args({25});
    b->Args({26});
    b->Args({27});
    b->Args({28});
    b->Args({29});
    b->Args({30});
    b->Args({31});
    b->Args({32});
    b->Args({33});
    b->Args({34});
    b->Args({35});
    b->Args({36});
    b->Args({37});
    b->Args({38});
    b->Args({39});
    b->Args({40});
    b->Args({41});
    b->Args({42});
    b->Args({43});
    b->Args({44});
    b->Args({45});
    b->Args({46});
    b->Args({47});
    b->Args({48});
    b->Args({49});
    b->Args({50});
    b->Args({51});
    b->Args({52});
    b->Args({53});
    b->Args({54});
    b->Args({55});
    b->Args({56});
    b->Args({57});
    b->Args({58});
    b->Args({59});
    b->Args({60});
    b->Args({61});
    b->Args({62});
    b->Args({63});
    b->Args({64});
    b->Args({65});
    b->Args({66});
    b->Args({67});
    b->Args({68});
}

BENCHMARK(Bench)->Apply(CustomArgs);

BENCHMARK_MAIN();