//
//  @author Loris Friedel
//

#include <chrono>
#include <iterator>
#include "../inc/MLPHand.hpp"

#include "opencv2/core.hpp"
#include "../inc/log.h"
#include "../inc/code.h"
#include "../inc/Timer.hpp"

namespace {
    const int BASE_LETTER = 'a';
}

MLPHand::MLPHand(int nbOfHiddenLayer, int nbOfNeuron) : networkPattern() {
    for (int i = 0; i < nbOfHiddenLayer; i++) {
        networkPattern.push_back(nbOfNeuron);
    }
}

MLPHand::MLPHand(const std::vector<int> networkPattern)
        : networkPattern(networkPattern) {}

int MLPHand::learnFrom(const std::string classifier_file_name) {
    model = cv::ml::StatModel::load<cv::ml::ANN_MLP>(classifier_file_name);

    LOG_I("Loading classifier...");
    if (model.empty()) {
        LOG_E("ERROR: Could not read the classifier : " << classifier_file_name);
        return Code::CASCADE_LOAD_ERROR;
    } else {
        cv::Mat layers = model.get()->getLayerSizes();
        networkPattern.clear();
        for (int i = 1; i < layers.rows - 1; i++) {
            networkPattern.push_back(model.get()->getLayerSizes().row(i).at<int>(0));
        }

        LOG_I("Classifier " << classifier_file_name << " successfully loaded!");
        return Code::SUCCESS;
    }
}

inline cv::TermCriteria MLPHand::TC(int iters, double eps) {
    return cv::TermCriteria(cv::TermCriteria::MAX_ITER + (eps > 0 ? cv::TermCriteria::EPS : 0), iters, eps);
}

int MLPHand::learnFrom(const cv::Mat &trainingData, const cv::Mat &trainingResponses) {
    Timer timeMonitor;

    // Start timer
    timeMonitor.start();

    const int nbOfLetters = 26;
    int nbOfSamples = trainingData.rows;

    cv::Mat formattedResponses = cv::Mat::zeros(nbOfSamples, nbOfLetters, CV_32FC1);

    // Unrolling the responses
    std::cout << "Formatting responses...";
    for (int i = 0; i < nbOfSamples; i++) {
        int cls_label = trainingResponses.at<int>(i) - BASE_LETTER;
        formattedResponses.at<float>(i, cls_label) = 1.f;
    }
    LOG_I(" done!");

    // Create and configure layers
    std::vector<int> layerSizes;
    layerSizes.push_back(trainingData.cols);
    for (int i = 0; i < networkPattern.size(); i++) {
        layerSizes.push_back(networkPattern[i]);
    }
    layerSizes.push_back(nbOfLetters);

    int method = cv::ml::ANN_MLP::BACKPROP;
    double methodEpsilon = 0.001;
    int maxIter = 128;

    // Train classifier
    cv::Ptr<cv::ml::TrainData> tData =
            cv::ml::TrainData::create(trainingData, cv::ml::ROW_SAMPLE, formattedResponses);

    // log
    std::stringstream patternStr;
    patternStr << "[";
    std::copy(layerSizes.begin(), layerSizes.end() - 1, std::ostream_iterator<int>(patternStr, ":"));
    patternStr << layerSizes.back() << "]";
    LOG_I("Training the classifier - layer pattern: " << patternStr.str() << " (may take a few minutes)...");

    model = cv::ml::ANN_MLP::create();
    model->setLayerSizes(layerSizes);
    model->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);
    model->setTermCriteria(TC(maxIter, 0));
    model->setTrainMethod(method, methodEpsilon);
    model->train(tData);

    // End timer
    timeMonitor.stop();

    LOG_I("Training done! (" << timeMonitor.getDurationS() << " s)");

    return Code::SUCCESS;
}

std::pair<int, float> MLPHand::predict(cv::Mat &img) {
    assert(model->isTrained());

    cv::Mat output;
    int result = (int) model->predict(img, output);
    return std::pair<int, float>(result, output.at<float>(0, result));
}

int MLPHand::exportModelTo(const std::string xml_file_name) {
    assert(model->isTrained());

    if (!xml_file_name.empty()) {
        LOG_I("Exporting model to " + xml_file_name);
        model->save(xml_file_name);
        LOG_I("Model successfully exported");
    } else {
        LOG_I("Error: model not exported");
        return Code::ERROR;
    }
}

double MLPHand::testOn(const cv::Mat &test_data, const cv::Mat &test_responses) {
    assert(model->isTrained());

    int nb_of_samples = test_data.rows;
    double total_success = 0;

    // Compute prediction error on test data
    // We count the number of prediction success
    for (int i = 0; i < nb_of_samples; i++) {
        total_success +=
                std::abs(model->predict(test_data.row(i)) + BASE_LETTER - test_responses.at<int>(i))
                <= FLT_EPSILON ? 1.f : 0.f;
    }

    double distance_average = total_success / nb_of_samples;

    return distance_average;
}