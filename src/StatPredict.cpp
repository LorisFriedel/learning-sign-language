//
// Created by loris on 11/17/16.
//

#include "../inc/StatPredict.hpp"

StatPredict::StatPredict(int label) : label(label) {}

StatPredict::~StatPredict() {
    for (TupleStat *stat : stats) {
        delete stat;
    }
}

void StatPredict::pushStat(const bool success, const int predictedLabel,
                           const float trustPercentage, const std::vector<float> predictOutput) {
    stats.push_back(new TupleStat(success, predictedLabel, trustPercentage, predictOutput));
}

const std::pair<int, int> StatPredict::successAndFailure() const {
    int success = 0;
    int failure = 0;
    for (TupleStat *stat : stats) {
        stat->success ? ++success : ++failure;
    }
    return {success, failure};
}

const std::pair<int, int> StatPredict::confusedLabel() const {
    // key: label that we thought it was and number of time confused as value
    std::map<int, int> confuseMap;
    for (TupleStat *stat : stats) {
        if (!stat->success) {
            // Check if already in the stat map
            if (confuseMap.find(stat->predictedLabel) == confuseMap.end()) {
                confuseMap[stat->predictedLabel] = 0;
            }
            confuseMap[stat->predictedLabel]++;
        }
    }

    // Find the maximum confusion label in the sub-map
    int maxNbOfConfusion = 0;
    int maxLabelConfused = 0;
    for (auto it = confuseMap.begin(); it != confuseMap.end(); ++it) {
        if (it->second > maxNbOfConfusion) {
            maxNbOfConfusion = it->second;
            maxLabelConfused = it->first;
        }
    }
    // total number of time we saw this label = stats.size();

    return {maxLabelConfused, maxNbOfConfusion};
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