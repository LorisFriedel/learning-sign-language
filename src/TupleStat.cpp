//
// @author Loris Friedel
//

#include "../inc/TupleStat.hpp"

TupleStat::TupleStat(const bool success, const int predictedLetter, const float trustPercentage,
                     const std::vector<float> &predictOutput)
        : success(success), predictedLetter(predictedLetter),
          trustPercentage(trustPercentage), predictOutput(predictOutput) {}