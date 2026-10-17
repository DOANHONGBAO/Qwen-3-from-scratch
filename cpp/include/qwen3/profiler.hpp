#pragma once

#include <bits/stdc++.h>
using namespace std;

// Bo do thoi gian rat don gian cho benchmark: cong don so giay theo ten.
// Mac dinh TAT (enabled = false) nen khong anh huong khi chay binh thuong.
//
//   {
//       ScopedTimer timer("attention.core");   // bat dau dem
//       ...                                    // doan code can do
//   }                                          // het khoi {} -> tu cong thoi gian
class Profiler {
public:
    static bool enabled;
    static map<string, double> seconds;
    static map<string, long long> calls;
    static void reset();
};

class ScopedTimer {
public:
    ScopedTimer(string name);
    ~ScopedTimer();   // destructor: tu chay khi bien ra khoi pham vi

private:
    string name_;
    chrono::steady_clock::time_point start_;
};
