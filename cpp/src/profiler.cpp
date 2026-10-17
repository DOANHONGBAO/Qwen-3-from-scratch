#include "qwen3/profiler.hpp"

bool Profiler::enabled = false;
map<string, double> Profiler::seconds;
map<string, long long> Profiler::calls;

void Profiler::reset() {
    seconds.clear();
    calls.clear();
}

ScopedTimer::ScopedTimer(string name) {
    if (!Profiler::enabled) return;
    name_ = name;
    start_ = chrono::steady_clock::now();
}

ScopedTimer::~ScopedTimer() {
    if (!Profiler::enabled || name_.empty()) return;
    double s = chrono::duration<double>(chrono::steady_clock::now() - start_).count();
    Profiler::seconds[name_] += s;
    Profiler::calls[name_]++;
}
