//
// Created by loris on 11/17/16.
//

#pragma once

#include <vector>
#include <map>
#include "TupleStat.hpp"

class StatPredict {
public:
    const int letterCode;
    std::vector<TupleStat *> stats;

    StatPredict(int letterCode);

    ~StatPredict();

    // Ajoute un test au total des informations  de resultat des prédictions
    void pushStat(const bool success, const int predictedLetter,
                  const float trustPercentage, const std::vector<float> predictOutput);

    // <0> nombre de succes, <1> nombre d'échec
    const std::pair<int, int> successAndFailure() const;

    // <0> La lettre la plus confondu (0 si aucune confusion), <1> nombre de fois confondu au total
    const std::pair<int, int> confuseLetter() const;

    // le taux de confiance min (0) / moyen (1) / max (2) pour les fois ou la lettre est correctement reconnue
    const std::tuple<double, double, double> trustWhenSuccess() const;
};