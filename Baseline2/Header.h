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

#define IMPORT_CONST(var) const float* var = static_cast<const float*>(__builtin_assume_aligned(_##var, 32))
#define IMPORT(var) float* var = static_cast<float*>(__builtin_assume_aligned(_##var, 32))

#include "Config.h"

#endif