//
//  @author Loris Friedel
//

#include <chrono>
#include "../inc/MLPHand.hpp"

#include "opencv2/core.hpp"
#include "../inc/log.h"
#include "../inc/code.h"
#include "../inc/Timer.hpp"

namespace {
    const int BASE_LETTER = 'a';
}

MLPHand::MLPHand(const int nb_of_hidden_layer, const int nb_of_neuron)
        : _nb_of_hidden_layer(nb_of_hidden_layer), _nb_of_neuron(nb_of_neuron) {}

int MLPHand::learn_from(const std::string classifier_file_name) {
    _model = cv::ml::StatModel::load<cv::ml::ANN_MLP>(classifier_file_name);

    LOG_I("Loading classifier...");
    if (_model.empty()) {
        LOG_E("ERROR: Could not read the classifier : " << classifier_file_name);
        return Code::CASCADE_LOAD_ERROR;
    } else {
        _nb_of_neuron = _model.get()->getLayerSizes().row(1).at<int>(0);
        _nb_of_hidden_layer = _model.get()->getLayerSizes().rows - 2;

        LOG_I("Classifier " << classifier_file_name << " successfully loaded!");
        return Code::SUCCESS;
    }
}

inline cv::TermCriteria MLPHand::TC(int iters, double eps) {
    return cv::TermCriteria(cv::TermCriteria::MAX_ITER + (eps > 0 ? cv::TermCriteria::EPS : 0), iters, eps);
}

int MLPHand::learn_from(const cv::Mat &training_data, const cv::Mat &training_responses) {
    Timer timeMonitor;

    // Start timer
    timeMonitor.start();

    const int nb_of_letters = 26;
    int nb_of_samples = training_data.rows;

    cv::Mat formatted_responses = cv::Mat::zeros(nb_of_samples, nb_of_letters, CV_32FC1);

    // Unrolling the responses
    LOG_I("Formatting responses...");
    for (int i = 0; i < nb_of_samples; i++) {
        int cls_label = training_responses.at<int>(i) - BASE_LETTER;
        formatted_responses.at<float>(i, cls_label) = 1.f;
    }

    // Create and configure layers
    int nb_of_layer = _nb_of_hidden_layer + 2;
    cv::Mat layer_sizes(nb_of_layer, 1, CV_16U);
    layer_sizes.row(0) = training_data.cols;
    for (int i = 1; i < nb_of_layer - 1; i++) {
        layer_sizes.row(i) = _nb_of_neuron;
    }
    layer_sizes.row(nb_of_layer - 1) = nb_of_letters;

    int method = cv::ml::ANN_MLP::BACKPROP;
    double method_epsilon = 0.001;
    int max_iter = 128;

    // Train classifier
    cv::Ptr<cv::ml::TrainData> t_data =
            cv::ml::TrainData::create(training_data, cv::ml::ROW_SAMPLE, formatted_responses);

    LOG_I("Training the classifier - " << _nb_of_hidden_layer << " layer of " << _nb_of_neuron
                                       << " neurons (may take a few minutes)...");
    _model = cv::ml::ANN_MLP::create();
    _model->setLayerSizes(layer_sizes);
    _model->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);
    _model->setTermCriteria(TC(max_iter, 0));
    _model->setTrainMethod(method, method_epsilon);
    _model->train(t_data);

    // End timer
    timeMonitor.stop();

    LOG_I("Training done! (" << timeMonitor.getDurationS() << " s)");

    return Code::SUCCESS;
}

std::pair<int, float> MLPHand::predict(cv::Mat &img) {
    assert(_model->isTrained());

    cv::Mat output;
    int result = (int) _model->predict(img, output);
    return std::pair<int, float>(result, output.at<float>(0, result));
}

void MLPHand::export_model_to(const std::string xml_file_name) {
    assert(_model->isTrained());

    if (!xml_file_name.empty()) {
        LOG_I("Exporting model to " + xml_file_name);
        _model->save(xml_file_name);
        LOG_I("Model successfully exported");
    }
}

double MLPHand::test_on(const cv::Mat &test_data, const cv::Mat &test_responses) {
    assert(_model->isTrained());

    int nb_of_samples = test_data.rows;
    double total_success = 0;

    // Compute prediction error on test data
    // We count the number of prediction success
    for (int i = 0; i < nb_of_samples; i++) {
        total_success +=
                std::abs(_model->predict(test_data.row(i)) + BASE_LETTER - test_responses.at<int>(i))
                <= FLT_EPSILON ? 1.f : 0.f;
    }

    double distance_average = total_success / nb_of_samples;

    return distance_average;
}