//
// @author Loris Friedel
//

#pragma once


#include <vector>

class TupleStat {
public:
    TupleStat(const bool success, const int predictedLetter,
              const float trustPercentage, const std::vector<float> &predictOutput);

    const bool success;
    const int predictedLetter;
    const float trustPercentage;
    const std::vector<float> predictOutput;
};