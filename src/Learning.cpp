//
// @author Loris Friedel
//

#include <random>
#include "../inc/Learning.hpp"
#include "../inc/log.h"
#include "../inc/code.h"
#include "../inc/Timer.hpp"
#include "../inc/DirectoryReader.hpp"
#include "../inc/DataYmlReader.hpp"

int trainMLPModel(cv::Mat &data, cv::Mat &responses,
                  MLPModel &model, const bool noTest, std::string testDir) {

    if (model.learnFrom(data, responses) == Code::SUCCESS) {
        if (!noTest) {
            return testModel(model, testDir);
        }
        return Code::SUCCESS;
    } else {
        LOGP_E(&model, "ERROR: model training failed");
        return Code::ERROR;
    }
}

int trainMLPModel(const std::string dataDir, const std::string testDir,
                  MLPModel &model, const bool noTest) {
    cv::Mat data;
    cv::Mat responses;

    LOGP_I(&model, "Start training process..");
    if (aggregateDataFrom(dataDir, data, responses) != Code::SUCCESS) {
        LOGP_E(&model, "Could not load training data");
        return Code::ERROR;
    };

    return trainMLPModel(data, responses, model, noTest, testDir);
}

int executeTestModel(std::string modelPath, std::string testDir, LabelMap &labelMap) {
    MLPModel model;
    model.setLabelMap(labelMap);
    model.learnFrom(modelPath);

    return testModel(model, testDir);
}

int testModel(MLPModel &model, cv::Mat &dataTest, cv::Mat &responsesTest) {
    LOGP_I(&model, "Testing model (" << dataTest.rows << " samples)...");
    std::pair<double, std::map<int, StatPredict *>> result = model.testOn(dataTest, responsesTest);
    LOGP_I(&model, "Testing done! " << std::endl << "Test result: " << result.first * 100 << "% success" << std::endl);

    for (auto it = result.second.begin(); it != result.second.end(); ++it) {
        int label = it->first;
        StatPredict &stat = *(it->second);

        std::pair<int, int> successFailure = stat.successAndFailure();
        std::pair<int, int> confusedLabel = stat.confusedLabel();
        std::tuple<double, double, double> trustValues = stat.trustWhenSuccess();
        LOGP_I(&model, "Label: " << model.convertLabel(label));
        LOGP_I(&model, " - Success: " << successFailure.first << "/" << stat.stats.size()
                             << " (" << (((double) successFailure.first / (double) stat.stats.size()) * 100)
                             << "% success rate)");
        LOGP_I(&model, " - Error: " << successFailure.second << "/" << stat.stats.size()
                           << " (" << (((double) successFailure.second / (double) stat.stats.size()) * 100)
                           << "% error rate)");
        if (confusedLabel.first != 0) {
            LOGP_I(&model, " - Most of the time confused with: " << model.convertLabel(confusedLabel.first)
                                                        << " ("
                                                        << (((double) confusedLabel.second /
                                                             (double) stat.stats.size()) * 100)
                                                        << "% of the time)");
        } else {
            LOGP_I(&model, " - No confusion with other labels");
        }

        LOGP_I(&model, " - Trust rate when success: ");
        LOGP_I(&model, " --> Minimum: " << std::get<0>(trustValues) * 100 << "%");
        LOGP_I(&model, " --> Average: " << std::get<1>(trustValues) * 100 << "%");
        LOGP_I(&model, " --> Maximum: " << std::get<2>(trustValues) * 100 << "%");
        LOGP_I(&model, "");

        delete it->second;
    }
    return Code::SUCCESS;
}

int testModel(MLPModel &model, std::string inputDir) {
    LOGP_I(&model, "Start testing process..");

    cv::Mat dataTest;
    cv::Mat responsesTest;

    if (aggregateDataFrom(inputDir, dataTest, responsesTest) != Code::SUCCESS) {
        LOGP_E(&model, "Could not load test data");
        return Code::ERROR;
    };

    return testModel(model, dataTest, responsesTest);
}

int aggregateDataFrom(std::string directory, cv::Mat &matData, cv::Mat &matResponses) {
    LOG_I("Loading data...");
    Timer timer;

    timer.start();
    DirectoryReader dirReader(directory);
    std::vector<std::string> dataPathList;
    int dirReadCode = dirReader.foreachFile([&dataPathList](std::string filePath, std::string fileName) {
        dataPathList.push_back(filePath);
    });

    if (dirReadCode == Code::SUCCESS) {
        auto engine = std::default_random_engine{};
        std::shuffle(std::begin(dataPathList), std::end(dataPathList), engine);
        for (std::string path : dataPathList) {
            int labelTmp;
            cv::Mat labelDataRow;

            // If no error while reading data
            DataYmlReader reader(path);
            if (reader.read(labelDataRow, labelTmp) != Code::SUCCESS) {
                labelDataRow.convertTo(labelDataRow, CV_32FC1);

                matResponses.push_back(labelTmp);
                matData.push_back(labelDataRow);
            } else {
                LOG_E("ERROR: can't load: " << path);
            }
        }
    } else {
        LOG_I("Loading data finished with errors.");
        return Code::ERROR;
    }
    timer.stop();

    LOG_I("Loading data done! (" << timer.getDurationS() << " s)");
    return Code::SUCCESS;
}
