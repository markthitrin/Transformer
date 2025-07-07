#include "cnpy.h"
#include <vector>
#include <cstring>
#include <cstdlib>

extern "C" {
    void load_npz_flat(const char file[], const char key[], int* out_len, float output[]) {
        cnpy::npz_t npz = cnpy::npz_load(file);
        if (npz.count(key) == 0) {
            *out_len = 0;
        }

        cnpy::NpyArray arr = npz[key];
        size_t total = 1;
        for (size_t s : arr.shape) total *= s;

        *out_len = total;

        memcpy(output, arr.data<float>(), sizeof(float) * total);
    }
}