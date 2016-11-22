// @author Loris Friedel

#include <tclap/CmdLine.h>
#include <cv.hpp>
#include <regex>
#include <random>
#include <thread>
#include "../inc/code.h"
#include "../inc/log.h"
#include "../inc/MLPHand.hpp"
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

        // TODO Add log redirection

        // For each data type
        std::vector<std::thread> learningThreads;

        for (std::string type : config.types) {
            using namespace std;
            using namespace cv;

            map<string, pair<Mat *, Mat *> *> datasetMap;

            // For each dataset
            for (string name : config.names) {
                LOG_I("Start thread for " << name << " data");
                learningThreads.push_back(std::thread(
                        [name, type, &datasetMap, &config]() {
                            // ON ANOTHER THREAD
                            using namespace std;
                            using namespace cv;

                            Mat *data = new Mat();
                            Mat *responses = new Mat();

                            stringstream trainDir;
                            trainDir << config.dataDir << "/" << name << "_" << type;

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
                                    // ON ANOTHER THREAD
                                    topoThreads.push_back(
                                            thread([topology, name, &data, &responses, &config, &type]() {
                                                MLPHand *model = new MLPHand(topology);
                                                LOGP_I(model, "Start thread for training " << topology << " on " << name
                                                                                           << " data");

                                                int learningCode = model->learnFrom(*data, *responses);
                                                if (learningCode == Code::SUCCESS) {

                                                    std::stringstream modelDir;
                                                    modelDir << config.modelDir << "/model_" << name << "_" << topology
                                                             << "_"
                                                             << type
                                                             << ".xml";

                                                    int exportCode = model->exportModelTo(modelDir.str());
                                                    if (exportCode != Code::SUCCESS) {
                                                        LOGP_E(model,
                                                               "ERROR: exporting " << modelDir.str() << " failed.");
                                                    }
                                                } else {
                                                    LOGP_E(model, "ERROR: training " << topology << " on " << name
                                                                                     << " data of type "
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

            // Delete everything
            for (auto &entry : datasetMap) {
                delete entry.second->first; // Delete input data
                delete entry.second->second; // Delete responses
                delete entry.second; // Delete pair
            }
        }

        return Code::SUCCESS;
    } catch (TCLAP::ArgException &e) {  // catch any exceptions
        LOG_E("error: " << e.error() << " for arg " << e.argId());
    }

    LOG_E("Program exited with errors");
    return Code::ERROR;
}