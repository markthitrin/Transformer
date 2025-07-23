#ifndef HEADER
#define HEADER


#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <immintrin.h>
#include <cstdlib>
#include <cmath>
#include <set>
#include <map>
#include <malloc.h>
#include <random>
#include <chrono>
#include <memory>
#include <cstring>
#include <cfloat>
#include <iomanip>
#include <chrono>
#include <omp.h>

static const int numPar = omp_get_max_threads();

inline int getNumThreads(const int N, const float tus) {
    return std::min(float(numPar), tus / 0.75f);
}

inline int getNumThreads(const int N, const float tus, const float flop, const float mem) {
    constexpr float fmratio = 4.8; // on 64 thread
    const float maxNumThreadMemCap = flop / mem / fmratio * 64;
    return std::min(std::min(float(numPar), tus / 0.75f), maxNumThreadMemCap);
}

#include "Config.h"

#endif