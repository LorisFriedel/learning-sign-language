//
// Created by loris on 11/17/16.
//

#pragma once

#include <vector>

class StatPredict {
public:
    StatPredict(int letterCode) : letterCode(letterCode) {}

    void addStat(bool success, int predictedLetter, float trustPercentage, std::vector<float> predictOutput) {
        //stats.push_back(std::tuple<bool, int, float, std::vector<float>>(success, predictedLetter, trustPercentage, predictOutput));
    }

    // pourcentage de succes sur cette lettre
    std::pair<int, int> succesAndFailed() {

    };

    // La lettre la plus confondu (+ pourcentage de fois confondu au total)
    std::pair<int, double> confuseLetter() {

    };

    // le taux de confiance min/moyen/max quand on reconnait bien la lettre
    std::tuple<double, double, double> trustWhenSucces() {

    };

private:
    int letterCode;
    std::vector<std::tuple<bool, int, float, std::vector<float>>> stats;
};