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

#define IMPORT_CONST(var) const float* var = static_cast<const float*>(_##var.data)
#define IMPORT(var) float* var = static_cast<float*>(_##var.data)
#define IMPORTA(name, var) float* name = static_cast<float*>(var.data)

// #define IMPORT_CONST(var) const float* var = static_cast<const float*>(__builtin_assume_aligned(_##var.data, 32))
// #define IMPORT(var) float* var = static_cast<float*>(__builtin_assume_aligned(_##var.data, 32))
// #define IMPORTM(class, var) float* var = static_cast<float*>(__builtin_assume_aligned(_##class.var.data, 32))


#include "Config.h"

#endif