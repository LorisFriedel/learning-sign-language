//
// @author Loris Friedel
//

#include "../inc/Timer.hpp"

using namespace std::chrono;

void Timer::start() {
    start_time = high_resolution_clock::now();
}

void Timer::stop() {
    end_time = high_resolution_clock::now();
}

double Timer::getDurationNS() {
    return duration_cast<nanoseconds>(end_time - start_time).count();;
}

double Timer::getDurationMS() {
    return duration_cast<nanoseconds>(end_time - start_time).count() / (1e6);;
}

double Timer::getDurationS() {
    return duration_cast<nanoseconds>(end_time - start_time).count() / (1e9);;
}
