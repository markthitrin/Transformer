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
    const float maxPar = std::min(numPar, N);
    return std::max(std::min(maxPar, tus / 0.75f), 1.f);
}

inline int getNumThreads(const int N, const float tus, const float flop, const float mem) {
    const float maxPar = std::min(numPar, N);
    constexpr float fmratio = 4.8; // on 64 thread
    const float maxNumThreadMemCap = flop / mem / fmratio * 64;
    return std::max(std::min(std::min(maxPar, tus / 0.75f), maxNumThreadMemCap), 1.f);
}

#include "Config.h"

#endif