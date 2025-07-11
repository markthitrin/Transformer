#include "Header.h"
#include "Timer.h"

void Timer::RestartRecord() {
    i = 0;
    t0 = std::chrono::steady_clock::now();
}

void Timer::CheckPoint() {
    std::chrono::duration<double, std::micro> duration = std::chrono::steady_clock::now() - t0;
    time[i].emplace_back(duration.count());
    maxI = std::max(maxI, ++i);
    t0 = std::chrono::steady_clock::now();
}

void Timer::Reserve(const int N) {
    for(int i = 0;i < timerSize;i++) {
        time[i].reserve(N);
    }
}

std::vector<double> Timer::GetTime() {
    std::vector<double> res(maxI);
    for(int i = 0;i < maxI;i++) {
        std::sort(time[i].begin(), time[i].end());
        int start = std::min(int(time[i].size() * 0.1), 20);
        int end = std::max(int((time[i].size() * 9 + 9) / 10), int(time[i].size()) - 20);
        double mean = 0.0;
        for(int j = start;j < end;j++) {
            mean += time[i][j];
        }
        mean /= end - start;
        
        res[i] = mean;
    }
    return res;
}


std::vector<double> Timer::time[timerSize];
std::chrono::time_point<std::chrono::steady_clock> Timer::t0;
int Timer::i;
int Timer::maxI;