//
// Created by loris on 11/17/16.
//

#pragma once

#include <vector>
#include <map>
#include "TupleStat.hpp"

class StatPredict {
public:
    const int label;
    std::vector<TupleStat *> stats;

    StatPredict(int label);

    ~StatPredict();

    // Ajoute un test au total des informations  de resultat des prédictions
    void pushStat(const bool success, const int predictedLabel,
                  const float trustPercentage, const std::vector<float> predictOutput);

    // <0> nombre de succes, <1> nombre d'échec
    const std::pair<int, int> successAndFailure() const;

    // <0> Le label le plus confondu (0 si aucune confusion), <1> nombre de fois confondu au total
    const std::pair<int, int> confusedLabel() const;

    // le taux de confiance min (0) / moyen (1) / max (2) pour les fois ou le label est correctement reconnue
    const std::tuple<double, double, double> trustWhenSuccess() const;
};