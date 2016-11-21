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

int trainMLPModel(const std::string dataDir, const std::string testDir,
                  MLPHand &model, const bool noTest) {
    cv::Mat data;
    cv::Mat responses;

    LOG_I("Start training process..");
    if (aggregateDataFrom(dataDir, data, responses) != Code::SUCCESS) {
        LOG_E("Could not load training data");
        return Code::ERROR;
    };

    if (model.learnFrom(data, responses) == Code::SUCCESS) {
        if (!noTest) {
            return testModel(model, testDir);
        }
        return Code::SUCCESS;
    } else {
        LOG_E("ERROR: model training failed");
        return Code::ERROR;
    }
}

int executeTestModel(std::string modelPath, std::string testDir) {
    MLPHand model;
    model.learnFrom(modelPath);

    return testModel(model, testDir);
}

int testModel(MLPHand &model, std::string inputDir) {
    LOG_I("Start testing process..");

    cv::Mat dataTest;
    cv::Mat responsesTest;

    if (aggregateDataFrom(inputDir, dataTest, responsesTest) != Code::SUCCESS) {
        LOG_E("Could not load test data");
        return Code::ERROR;
    };

    std::cout << "Testing model (" << dataTest.rows << " samples)...";
    std::cout.flush();
    std::pair<double, std::map<int, StatPredict *>> result = model.testOn(dataTest, responsesTest);
    LOG_I(" done! " << std::endl << "Test result: " << result.first * 100 << "% success" << std::endl);

    for (auto it = result.second.begin(); it != result.second.end(); ++it) {
        int letterCode = it->first;
        StatPredict &stat = *(it->second);

        std::pair<int, int> successFailure = stat.successAndFailure();
        std::pair<int, int> confuseLetter = stat.confuseLetter();
        std::tuple<double, double, double> trustValues = stat.trustWhenSuccess();
        LOG_I("Letter: " << std::string(1, letterCode));
        LOG_I(" - Success: " << successFailure.first << "/" << stat.stats.size()
                             << " (" << (((double) successFailure.first / (double) stat.stats.size()) * 100)
                             << "% success rate)");
        LOG_I(" - Error: " << successFailure.second << "/" << stat.stats.size()
                           << " (" << (((double) successFailure.second / (double) stat.stats.size()) * 100)
                           << "% error rate)");
        if(confuseLetter.first != 0) {
            LOG_I(" - Most of the time confused with: " << std::string(1, confuseLetter.first)
                                                        << " ("
                                                        << (((double) confuseLetter.second / (double) stat.stats.size()) * 100)
                                                        << "% of the time)");
        } else {
            LOG_I(" - No confusion with other letters");
        }

        LOG_I(" - Trust rate when success: ");
        LOG_I(" --> Minimum: " << std::get<0>(trustValues)*100 << "%");
        LOG_I(" --> Average: " << std::get<1>(trustValues)*100 << "%");
        LOG_I(" --> Maximum: " << std::get<2>(trustValues)*100 << "%");
        LOG_I("");

        delete it->second;
    }
    return Code::SUCCESS;
}

int aggregateDataFrom(std::string directory, cv::Mat &matData, cv::Mat &matResponses) {
    std::cout << "Loading data..."; std::cout.flush();
    Timer timer;

    timer.start();
    DirectoryReader dirReader(directory);
    std::vector<std::string> dataPathList;
    int dirReadCode = dirReader.foreachFile([&dataPathList](std::string filePath, std::string fileName) {
        dataPathList.push_back(filePath);
    });

    if(dirReadCode == Code::SUCCESS) {
        auto engine = std::default_random_engine{};
        std::shuffle(std::begin(dataPathList), std::end(dataPathList), engine);
        for(std::string path : dataPathList) {
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
        LOG_I(" finished with errors.");
        return Code::ERROR;
    }
    timer.stop();

    LOG_I(" done! (" << timer.getDurationS() << " s)");
    return Code::SUCCESS;
}