//
//  @author Loris Friedel
//

#pragma once

#include <chrono>
#include <iomanip>

std::string current_date_time() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%m_%d-%H_%M_%S");
    return ss.str();
}

long get_timestamp() {
    return std::chrono::system_clock::now().time_since_epoch().count();
}