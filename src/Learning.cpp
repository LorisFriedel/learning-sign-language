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
                  MLPHand &model, const bool noTest, std::string testDir) {

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
                  MLPHand &model, const bool noTest) {
    cv::Mat data;
    cv::Mat responses;

    LOGP_I(&model, "Start training process..");
    if (aggregateDataFrom(dataDir, data, responses) != Code::SUCCESS) {
        LOGP_E(&model, "Could not load training data");
        return Code::ERROR;
    };

    return trainMLPModel(data, responses, model, noTest, testDir);
}

int executeTestModel(std::string modelPath, std::string testDir) {
    MLPHand model;
    model.learnFrom(modelPath);

    return testModel(model, testDir);
}

int testModel(MLPHand &model, cv::Mat &dataTest, cv::Mat &responsesTest) {
    LOGP_I(&model, "Testing model (" << dataTest.rows << " samples)...");
    std::pair<double, std::map<int, StatPredict *>> result = model.testOn(dataTest, responsesTest);
    LOGP_I(&model, "Testing done! " << std::endl << "Test result: " << result.first * 100 << "% success" << std::endl);

    for (auto it = result.second.begin(); it != result.second.end(); ++it) {
        int letterCode = it->first;
        StatPredict &stat = *(it->second);

        std::pair<int, int> successFailure = stat.successAndFailure();
        std::pair<int, int> confuseLetter = stat.confuseLetter();
        std::tuple<double, double, double> trustValues = stat.trustWhenSuccess();
        LOGP_I(&model, "Letter: " << std::string(1, letterCode));
        LOGP_I(&model, " - Success: " << successFailure.first << "/" << stat.stats.size()
                             << " (" << (((double) successFailure.first / (double) stat.stats.size()) * 100)
                             << "% success rate)");
        LOGP_I(&model, " - Error: " << successFailure.second << "/" << stat.stats.size()
                           << " (" << (((double) successFailure.second / (double) stat.stats.size()) * 100)
                           << "% error rate)");
        if (confuseLetter.first != 0) {
            LOGP_I(&model, " - Most of the time confused with: " << std::string(1, confuseLetter.first)
                                                        << " ("
                                                        << (((double) confuseLetter.second /
                                                             (double) stat.stats.size()) * 100)
                                                        << "% of the time)");
        } else {
            LOGP_I(&model, " - No confusion with other letters");
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

int testModel(MLPHand &model, std::string inputDir) {
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
            int letterTmp;
            cv::Mat letterDataRow;

            // If no error while reading data
            DataYmlReader reader(path);
            if (reader.read(letterDataRow, letterTmp) != Code::SUCCESS) {
                letterDataRow.convertTo(letterDataRow, CV_32FC1);

                matResponses.push_back(letterTmp);
                matData.push_back(letterDataRow);
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
