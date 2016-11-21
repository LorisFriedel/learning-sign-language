//
// @author Loris Friedel
//

#pragma once

#include <string>
#include "MLPHand.hpp"

int trainMLPModel(const std::string dataDir, const std::string testDir,
                  MLPHand &model, const bool noTest);

int aggregateDataFrom(std::string directory, cv::Mat &matData, cv::Mat &matResponses);

int executeTestModel(std::string modelPath, std::string testDir);

int testModel(MLPHand &model, std::string inputDir);
