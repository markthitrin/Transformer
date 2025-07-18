#ifndef TIMER
#define TIMER

#include "Header.h"

constexpr int timerSize = 10000;

class Timer {
public:
    static void RestartRecord();
    static void CheckPoint();
    static void Reserve(const int N);
    static std::vector<double> GetTime();
    static std::vector<double> GetTimeStd();

    static std::vector<double> time[timerSize];
    static std::chrono::time_point<std::chrono::steady_clock> t0;
    static int i;
    static int maxI;
};

#endif