//
// @author Loris Friedel
//

#include <tclap/CmdLine.h>
#include <cv.hpp>
#include <regex>
#include <random>
#include "../inc/code.h"
#include "../inc/log.h"
#include "../inc/constant.h"
#include "../inc/MLPModel.hpp"
#include "../inc/time.h"
#include "../inc/Learning.hpp"

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
                                                "Specify a directory where .yml file are located. Those files will be used to test the model. Default value is " +
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

        TCLAP::ValueArg<std::string> topologyArg("p", "intern-topology",
                                                       "Define the network pattern (hidden layers topology). Example: '4 2 4' define a 3 layers network, with 4 neurons for the first one, 2 for the second and 3 for the last one.",
                                                       false, Default::TOPOLOGY, "topology", cmd);

        TCLAP::SwitchArg noTestArg("s", "skip-test",
                                   "Skip the model test.",
                                   cmd, false);

        TCLAP::SwitchArg testOnlyArg("y", "test-only",
                                     "If this argument is present, the program will only test the model specified by the '--model-to-test' arg. If '--model-to-test' is not specified, the program exit.",
                                     cmd, false);

        TCLAP::ValueArg<std::string> jsonDistribArg("d", "distribution-output",
                                                "Specify the path to a JSON file where to write label distribution of training data (create it if not exists).",
                                                false, "", "pathToJsonFile", cmd);

        // TODO add an input file in args for label mapping

        //// Parse the argv array
        cmd.parse(argc, argv);

        //// Get the value parsed by each arg and handle them

        std::string &testDir = testDirArg.getValue();
        std::string &jsonDistribPath = jsonDistribArg.getValue();

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

            std::string &model_to_test = modelInputArg.getValue();

            return executeTestModel(model_to_test, testDir);

        } else { // Learning mode
            std::string &modelOutPath = modelOutputArg.getValue();
            std::string &dataDir = dataDirArg.getValue();
            bool noTest = noTestArg.getValue();

            MLPModel model;

            if (topologyArg.isSet()) {
                std::string &topology = topologyArg.getValue();
                model = MLPModel(topology);
            } else {
                int nbOfLayer = nbLayerArg.getValue();
                int nbOfNeuron = nbNeuronArg.getValue();
                model = MLPModel(nbOfLayer, nbOfNeuron);
            }

            if (trainMLPModel(dataDir, testDir, model, noTest) == Code::SUCCESS) {
                model.exportTrainDataDistribution(jsonDistribPath);
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