#include <iostream>
#include <omp.h>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <numeric>

static void escape(void *p) {
    asm volatile("" : : "g"(p) : "memory");
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: ./array_add_benchmark <array_size>\n";
        return 1;
    }

    const int N_RUNS = 10;
    const size_t size = std::stoul(argv[1]);

    std::vector<float> A(size, 1.0f);
    std::vector<float> B(size, 2.0f);
    std::vector<float> C(size, 0.0f);

    int max_threads = omp_get_max_threads();

    std::cout << "Array size: " << size << "\n";
    std::cout << "Testing 1 to " << max_threads << " threads:\n";

    for (int num_threads = 1; num_threads <= max_threads; ++num_threads) {
        omp_set_num_threads(num_threads);
        double total_time = 0.0;

        for (int run = 0; run < N_RUNS; ++run) {
            double start = omp_get_wtime();

            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < num_threads; ++i)
                for(int  j= 0;j < 1;j++) {
                    C[i] = A[i] + B[i];
                    escape(&C[i]);
                }

            double end = omp_get_wtime();
            escape(C.data());
            total_time += (end - start);
        }

        double mean_time = total_time / N_RUNS;
        // std::cout << "Threads: " << num_threads
        //           << " | Mean Time: " << mean_time * 1000 << " ms\n";
        std::cout << 1000.0 / mean_time << std::endl;
    }

    return 0;
}