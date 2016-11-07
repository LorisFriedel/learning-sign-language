//
// @author Loris Friedel
//

#include <tclap/CmdLine.h>
#include <cv.hpp>
#include <dirent.h>
#include "../inc/code.h"
#include "../inc/log.h"
#include "../inc/constant.h"
#include "../inc/MLPHand.hpp"
#include "../inc/time.h"

int generate_MLP_model(const std::string model_path, const std::string data_dir, const std::string test_dir,
                       const int nb_of_layer, const int nb_of_neuron, const bool no_test);

int aggregate_data_from(std::string directory, cv::Mat &mat_data, cv::Mat &mat_responses);

int execute_test_model(std::string model_path, std::string test_dir);

double test_model(MLPHand &model, std::string input_dir);

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

        TCLAP::SwitchArg noTestArg("s", "no-test",
                                   "Skip the model test.",
                                   cmd, false);

        TCLAP::SwitchArg testOnlyArg("y", "test-only",
                                     "If this arg is present, the program will only test the model specified by the '--model-to-test' arg. If '--model-to-test' is not specified, the program exit.",
                                     cmd, false);

        //// Parse the argv array
        cmd.parse(argc, argv);

        //// Get the value parsed by each arg and handle them

        std::string test_dir = testDirArg.getValue();

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

            return execute_test_model(model_to_test, test_dir);

        } else { // Learning mode
            std::string model_out_path = modelOutputArg.getValue();
            std::string data_dir = dataDirArg.getValue();
            int nb_of_layer = nbLayerArg.getValue();
            int nb_of_neuron = nbNeuronArg.getValue();
            bool no_test = noTestArg.getValue();

            return generate_MLP_model(model_out_path, data_dir, test_dir, nb_of_layer, nb_of_neuron, no_test);
        }

    } catch (TCLAP::ArgException &e) {  // catch any exceptions
        LOG_E("error: " << e.error() << " for arg " << e.argId());
    }

    LOG_E("Program exited with errors");
    return Code::ERROR;
}


int generate_MLP_model(const std::string model_path, const std::string data_dir, const std::string test_dir,
                       const int nb_of_layer, const int nb_of_neuron, const bool no_test) {
    cv::Mat data;
    cv::Mat responses;

    if (aggregate_data_from(data_dir, data, responses) != Code::SUCCESS) {
        LOG_E("Cant load training data");
        return Code::ERROR;
    };

    MLPHand model(nb_of_layer, nb_of_neuron);

    if(model.learn_from(data, responses) == Code::SUCCESS) {
        model.export_model_to(model_path);

        if (!no_test) {
            double test_result = test_model(model, test_dir);
            LOG_I("Test MLP over test data set... : " << test_result * 100 << "% success");
        }

        return Code::SUCCESS;
    } else {
        return Code::ERROR;
    }
}

int execute_test_model(std::string model_path, std::string test_dir) {
    MLPHand model;
    model.learn_from(model_path);

    LOG_I("Testing model...");
    double test_result = test_model(model, test_dir);
    LOG_I("Test result: " << test_result * 100 << "% success");
}

double test_model(MLPHand &model, std::string input_dir) {
    cv::Mat data_test;
    cv::Mat responses_test;

    if (aggregate_data_from(input_dir, data_test, responses_test) != Code::SUCCESS) {
        LOG_E("Cant load test data");
        return Code::ERROR;
    };

    return model.test_on(data_test, responses_test);
}

int aggregate_data_from(std::string directory, cv::Mat &mat_data, cv::Mat &mat_responses) {
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(directory.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            if (ent->d_type == DT_REG) {
                std::string file_path = ent->d_name;

                int letter_tmp;
                cv::Mat letter_data_row;

                std::stringstream full_path;
                full_path << directory << "/" << file_path;
                cv::FileStorage file(full_path.str(), cv::FileStorage::READ);

                // If no error while reading data
                if(MLPHand::read_data(file, letter_data_row, letter_tmp) != Code::SUCCESS) {
                    letter_data_row.convertTo(letter_data_row, CV_32FC1, 1.0 / 255.0); //TODO maybe not for HOG ?
                    file.release();

                    mat_responses.push_back(letter_tmp);
                    mat_data.push_back(letter_data_row);
                }
            }
        }
        closedir(dir);
    } else {
        // Could not open directory
        return Code::ERROR;
    }
    return Code::SUCCESS;
}