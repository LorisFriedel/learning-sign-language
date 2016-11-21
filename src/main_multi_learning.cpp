//
// @author Loris Friedel
//

#include <tclap/CmdLine.h>
#include <cv.hpp>
#include <regex>
#include <random>
#include <thread>
#include "../inc/code.h"
#include "../inc/log.h"
#include "../inc/MLPHand.hpp"
#include "../inc/time.h"
#include "../inc/MultiConfig.hpp"
#include "../inc/Learning.hpp"

int main(int argc, const char **argv) {
    try {
        TCLAP::CmdLine cmd(
                "!!! Help for sign language multi-threaded learning program. !!!"
                        "\nWritten by Loris Friedel",
                ' ', "1.0");

        TCLAP::ValueArg<std::string> configFileArg("c", "config",
                                                   "Configuration file. See multi_learning_config_example.yml for details.",
                                                   false, "./multi_learning_config_example.yml",
                                                   "PATH_TO_JSON_CONFIG_FILE", cmd);


        //// Parse the argv array
        cmd.parse(argc, argv);

        //// Get the value parsed by each arg and handle them
        std::string configPath = configFileArg.getValue();
        MultiConfig config(configPath);

        // TODO add log redirection !!!!!!

         // TODO !!!!!!

        // TODO rewrite !!!!

        // TODO !!!!

        // For each data type
        std::vector<std::thread> learningThreads;

        for (std::string type : config.types) {
            using namespace std;
            using namespace cv;

            map<string, pair<Mat *, Mat *> *> datasetMap;
            map<string, map<string, MLPHand *> *> modelMap;



            // For each dataset
            for (string name : config.names) {
                learningThreads.push_back(std::thread(
                        [&]() {
                            // ON ANOTHER THREAD
                            using namespace std;
                            using namespace cv;
                            Mat *data = new Mat();
                            Mat *responses = new Mat();

                            stringstream trainDir;
                            trainDir << config.trainDir << "/" << name << "_" << type;

                            // Load data just in time
                            int loadDataCode = aggregateDataFrom(trainDir.str(), *data, *responses);
                            if (loadDataCode != Code::SUCCESS) {
                                LOG_E("ERROR: Could not load training data \"" << trainDir.str() << "\"");
                                delete data;
                                delete responses;
                            } else {
                                datasetMap[name] = new pair<Mat *, Mat *>(data, responses);

                                vector<thread> topoThreads;
                                // For each topology, train a model
                                for (string topology : config.topologies) {
                                    MLPHand *model = new MLPHand(topology);

                                    if (modelMap.find(name) == modelMap.end()) {
                                        modelMap[name] = new map<string, MLPHand *>();
                                    }
                                    map<string, MLPHand *> &modelMapName = *modelMap[name];
                                    modelMapName[topology] = model;

                                    LOG_I("Start training process..");

                                    // ON ANOTHER THREAD
                                    topoThreads.push_back(thread([&]() {
                                        int learningCode = model->learnFrom(*data, *responses);
                                        if (learningCode == Code::SUCCESS) {

                                            std::stringstream modelDir;
                                            modelDir << config.modelDir << "/model_" << name << "_" << topology << "_"
                                                     << type
                                                     << ".xml";

                                            int exportCode = model->exportModelTo(modelDir.str());
                                            if (exportCode != Code::SUCCESS) {
                                                LOG_E("ERROR: exporting " << modelDir.str() << " failed.");
                                            }
                                        } else {
                                            LOG_E("ERROR: training " << topology << " on " << name << " data of type "
                                                                     << type
                                                                     << " failed.");
                                        }
                                    }));
                                    // END: ON ANOTHER THREAD
                                }

                                for (std::thread &t : topoThreads) {
                                    t.join();
                                };
                            }
                            // END: ON ANOTHER THREAD
                        }
                ));
            }

            for (std::thread &t : learningThreads) {
                t.join();
            };


            std::vector<std::thread> testThreads;
            // test each topology of model 'modelTest' on testData dataset
            for (std::string modelTest : config.names) {
                for (std::string testData : config.names) {
                    // ON ANOTHER THREAD
                    testThreads.push_back(std::thread([&]() {
                        for (std::string topology : config.topologies) {
                            std::map<std::string, MLPHand *> &modelMapModelTest = *modelMap[modelTest];
                            MLPHand *model = modelMapModelTest[topology];

                            std::stringstream testDir;
                            testDir << config.testDir << "/" << testData << "_" << type;

                            int testCode = testModel(*model, testDir.str()); // TODO use dataset already loaded !
                            if (testCode != Code::SUCCESS) {
                                LOG_E("ERROR: error while testing model " << modelTest << "_" << topology
                                                                          << " on \"" << testDir.str() << "\" dataset");
                            }
                        }
                    }));
                    // END: ON ANOTHER THREAD
                }
            }

            for (std::thread &t : testThreads) {
                t.join();
            };
        }

        return Code::SUCCESS;
    } catch (TCLAP::ArgException &e) {  // catch any exceptions
        LOG_E("error: " << e.error() << " for arg " << e.argId());
    }

    LOG_E("Program exited with errors");
    return Code::ERROR;
}