//
// @author Loris Friedel
//

#pragma once

#include <chrono>

class Timer {

public:
    void start();

    void stop();

    double getDurationNS();

    double getDurationMS();

    double getDurationS();

private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;

};

