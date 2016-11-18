//
// Created by loris on 11/17/16.
//

#include "../inc/StatPredict.hpp"

StatPredict::StatPredict(int letterCode) : letterCode(letterCode) {}

StatPredict::~StatPredict() {
    for (TupleStat *stat : stats) {
        delete stat;
    }
}

void StatPredict::pushStat(const bool success, const int predictedLetter,
                           const float trustPercentage, const std::vector<float> predictOutput) {
    stats.push_back(new TupleStat(success, predictedLetter, trustPercentage, predictOutput));
}

const std::pair<int, int> StatPredict::successAndFailure() const {
    int success = 0;
    int failure = 0;
    for (TupleStat *stat : stats) {
        stat->success ? ++success : ++failure;
    }
    return {success, failure};
}

const std::pair<int, int> StatPredict::confuseLetter() const {
    // key: letter that we thought it was and number of time confused as value
    std::map<int, int> confuseMap;
    for (TupleStat *stat : stats) {
        if (!stat->success) {
            // Check if already in the stat map
            if (confuseMap.find(stat->predictedLetter) == confuseMap.end()) {
                confuseMap[stat->predictedLetter] = 0;
            }
            confuseMap[stat->predictedLetter]++;
        }
    }

    // Find the maximum confusion letter in the sub-map
    int maxNbOfConfusion = 0;
    int maxLetterConfused = 0;
    for (auto it = confuseMap.begin(); it != confuseMap.end(); ++it) {
        if (it->second > maxNbOfConfusion) {
            maxNbOfConfusion = it->second;
            maxLetterConfused = it->first;
        }
    }
    // total number of time we saw this letter = stats.size();

    return {maxLetterConfused, maxNbOfConfusion};
}

const std::tuple<double, double, double> StatPredict::trustWhenSuccess() const {
    double nbOfTime = 0.0;
    double min = 1.1;
    double total = 0.0;
    double max = -1.1;
    for (TupleStat *stat : stats) {
        if (stat->success) {
            if (stat->trustPercentage < min) {
                min = stat->trustPercentage;
            }
            if (stat->trustPercentage > max) {
                max = stat->trustPercentage;
            }
            total += stat->trustPercentage;
            ++nbOfTime;
        }
    }

    if (nbOfTime != 0.0) {
        return {min, total / nbOfTime, max};
    }
    return {0, 0, 0};
};