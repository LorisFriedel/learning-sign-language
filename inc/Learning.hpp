//
// @author Loris Friedel
//

#pragma once

#include <string>
#include "MLPModel.hpp"

int trainMLPModel(cv::Mat &data, cv::Mat &responses,
                  MLPModel &model, const bool noTest = true, std::string testDir = "");

int trainMLPModel(const std::string dataDir, const std::string testDir,
                  MLPModel &model, const bool noTest);

int aggregateDataFrom(std::string directory, cv::Mat &matData, cv::Mat &matResponses);

int executeTestModel(std::string modelPath, std::string testDir);

int testModel(MLPModel &model, cv::Mat &dataTest, cv::Mat &responsesTest);

int testModel(MLPModel &model, std::string inputDir);
