//
// @author Loris Friedel
//

#include <tclap/CmdLine.h>
#include <cv.hpp>
#include <dirent.h>
#include <regex>
#include <random>
#include "../inc/code.h"
#include "../inc/log.h"
#include "../inc/constant.h"
#include "../inc/MLPHand.hpp"
#include "../inc/time.h"
#include "../inc/DataYmlReader.hpp"
#include "../inc/DirectoryReader.hpp"
#include "../inc/Timer.hpp"

int trainMLPModel(const std::string dataDir, const std::string testDir,
                  MLPHand &model, const bool noTest);

int aggregateDataFrom(std::string directory, cv::Mat &matData, cv::Mat &matResponses);

int executeTestModel(std::string modelPath, std::string testDir);

int testModel(MLPHand &model, std::string inputDir);

std::vector<int> parsePattern(std::string basic_string);

int main(int argc, const char **argv) {
    try {
        TCLAP::CmdLine cmd(
                "!!! Help for sign language learning program. !!!"
                        "\nUsage examples:"
                        "\n./learning.exe -o model_v1.xml -l 4 -n 32 -i images/data/learn -t images/data/test"
                        "\n -- This execution will generate a model named model_v1.xml in the current directory, with 4 layer of 32 neurons using data 'images/data/learn' to learn and '/images/data/test' to test the model"
                        "\n./learning.exe --test-only -m model_v1.xml -t images/data/test"
                        "\n -- This execution will test the model named model_v1.xml over the data set located in the '/images/data/test' directory"
                        "\nWritten by Loris Friedel",
                ' ', "1.0");

        TCLAP::ValueArg<std::string> modelOutputArg("o", "output",
                                                    "Specify the path where to save the generated model. Default value is " +
                                                    Default::MODEL_PATH,
                                                    false, Default::MODEL_PATH, "PATH_TO_XML_MODEL_FILE", cmd);

        TCLAP::ValueArg<std::string> modelInputArg("m", "model-to-test",
                                                   "Specify the model to use for test only. If --test is not specified, this argument is not used and the program exit. Default value is " +
                                                   Default::MODEL_PATH,
                                                   false, Default::MODEL_PATH, "PATH_TO_XML_MODEL_FILE", cmd);

        TCLAP::ValueArg<std::string> dataDirArg("i", "data-dir",
                                                "Specify a directory where .yml file are located. Those files will be used to train the model. Default value is " +
                                                Default::LETTERS_DATA_PATH,
                                                false, Default::LETTERS_DATA_PATH, "DIRECTORY_PATH", cmd);

        TCLAP::ValueArg<std::string> testDirArg("t", "test_dir",
                                                "Specify a directory where .yml file are located. Those files will be used to test the model. Defaul value is " +
                                                Default::LETTERS_DATA_PATH,
                                                false, Default::LETTERS_DATA_PATH, "DIRECTORY_PATH", cmd);

        TCLAP::ValueArg<int> nbLayerArg("l", "layer",
                                        "Specify the number of layer for the model. Default value is " +
                                        std::to_string(Default::NB_OF_LAYER),
                                        false, Default::NB_OF_LAYER, "POSITIVE_INTEGER", cmd);

        TCLAP::ValueArg<int> nbNeuronArg("n", "neuron",
                                         "Specify the number of neuron per layer for the model. Default value is " +
                                         std::to_string(Default::NB_OF_NEURON),
                                         false, Default::NB_OF_NEURON, "POSITIVE_INTEGER", cmd);

        TCLAP::ValueArg<std::string> networkPatternArg("p", "network-pattern",
                                                       "Define the network pattern. Example: '4 2 4' define a 3 layers network, with 4 neurons for the first one, 2 for the second and 3 for the last one.",
                                                       false, "64 64", "pattern", cmd);

        TCLAP::SwitchArg noTestArg("s", "no-test",
                                   "Skip the model test.",
                                   cmd, false);

        TCLAP::SwitchArg testOnlyArg("y", "test-only",
                                     "If this arg is present, the program will only test the model specified by the '--model-to-test' arg. If '--model-to-test' is not specified, the program exit.",
                                     cmd, false);

        //// Parse the argv array
        cmd.parse(argc, argv);

        //// Get the value parsed by each arg and handle them

        std::string testDir = testDirArg.getValue();

        // Test mode
        if (testOnlyArg.isSet() || modelInputArg.isSet()) {
            if (!testOnlyArg.isSet()) {
                LOG_E("You must specify the '--test-only' arg");
                return Code::ERROR;
            }
            if (!modelInputArg.isSet()) {
                LOG_E("You must specify the '--model-to-test' arg");
                return Code::ERROR;
            }

            std::string model_to_test = modelInputArg.getValue();

            return executeTestModel(model_to_test, testDir);

        } else { // Learning mode
            std::string modelOutPath = modelOutputArg.getValue();
            std::string dataDir = dataDirArg.getValue();
            bool noTest = noTestArg.getValue();

            MLPHand model;

            if (networkPatternArg.isSet()) {
                std::string pattern = networkPatternArg.getValue();
                std::vector<int> networkPattern = parsePattern(pattern);
                if (networkPattern.empty()) {
                    return Code::ERROR;
                } else {
                    model = MLPHand(networkPattern);
                }
            } else {
                int nbOfLayer = nbLayerArg.getValue();
                int nbOfNeuron = nbNeuronArg.getValue();
                model = MLPHand(nbOfLayer, nbOfNeuron);
            }

            if (trainMLPModel(dataDir, testDir, model, noTest) == Code::SUCCESS) {
                return model.exportModelTo(modelOutPath);
            } else {
                return Code::ERROR;
            }
        }
    } catch (TCLAP::ArgException &e) {  // catch any exceptions
        LOG_E("error: " << e.error() << " for arg " << e.argId());
    }

    LOG_E("Program exited with errors");
    return Code::ERROR;
}

std::vector<int> parsePattern(std::string pattern) {
    std::vector<int> result;

    std::stringstream ss(pattern);
    std::string tok;
    char delimiter = ' ';

    while (getline(ss, tok, delimiter)) {
        result.push_back(std::stoi(tok));
    }

    return result;
}


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
    int dirReadSuccess = dirReader.foreachFile([&dataPathList](std::string filePath, std::string fileName) {
        dataPathList.push_back(filePath);
    });

    if(dirReadSuccess) {
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