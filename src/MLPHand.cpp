//
//  @author Loris Friedel
//

#include <chrono>
#include <iterator>
#include "../inc/MLPHand.hpp"
#include "../inc/log.h"
#include "../inc/code.h"
#include "../inc/Timer.hpp"

namespace {
    const int BASE_LETTER = 'a';
}

MLPHand::MLPHand(int nbOfHiddenLayer, int nbOfNeuron) {
    for (int i = 0; i < nbOfHiddenLayer; i++) {
        networkPattern.push_back(nbOfNeuron);
    }
}

MLPHand::MLPHand(const std::vector<int> networkPattern)
        : networkPattern(networkPattern) {}

MLPHand::MLPHand(const std::string networkPatternStr) {
    std::stringstream ss(networkPatternStr);
    std::string tok;

    char delimiter = networkPatternStr.find('_') != std::string::npos ? '_' : ' ';

    while (getline(ss, tok, delimiter)) {
        networkPattern.push_back(std::stoi(tok));
    }
}

int MLPHand::learnFrom(const std::string classifier_file_name) {
    model = cv::ml::StatModel::load<cv::ml::ANN_MLP>(classifier_file_name);

    LOGP_I(this, "Loading classifier...");
    if (model.empty()) {
        LOGP_E(this, "ERROR: Could not read the classifier : " << classifier_file_name);
        return Code::CASCADE_LOAD_ERROR;
    } else {
        cv::Mat layers = model.get()->getLayerSizes();
        networkPattern.clear();
        for (int i = 1; i < layers.rows - 1; i++) {
            networkPattern.push_back(model.get()->getLayerSizes().row(i).at<int>(0));
        }

        LOGP_I(this, "Classifier " << classifier_file_name << " successfully loaded!");
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
    LOGP_I(this, "Formatting responses...");
    std::map<int, int> lettersCountMap;
    for (int i = 0; i < nbOfSamples; i++) {
        int responseLetter = trainingResponses.at<int>(i);
        int clsLabel =  responseLetter - BASE_LETTER;
        formattedResponses.at<float>(i, clsLabel) = 1.f;

        if(lettersCountMap.find(responseLetter) == lettersCountMap.end()) {
            lettersCountMap[responseLetter] = 0;
        }
        lettersCountMap[responseLetter]++;
    }
    LOGP_I(this, "Formatting responses done!");

    LOGP_I(this, "Training samples composition: ");
    for (auto it = lettersCountMap.begin(); it != lettersCountMap.end(); ++it) {
        LOGP_I(this, " - " << std::string(1, it->first) << " * " << it->second);
    }
    LOGP_I(this, "");

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
    LOGP_I(this, "Training the classifier (" << nbOfSamples << " samples) - layer pattern: " << patternStr.str()
                                      << " (may take a few minutes)...");

    model = cv::ml::ANN_MLP::create();
    model->setLayerSizes(layerSizes);
    model->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);
    model->setTermCriteria(TC(maxIter, 0));
    model->setTrainMethod(method, methodEpsilon);
    model->train(tData);

    // End timer
    timeMonitor.stop();

    LOGP_I(this, "Training done! (" << timeMonitor.getDurationS() << " s)");

    return Code::SUCCESS;
}

std::pair<int, float> MLPHand::predict(cv::Mat &img) {
    assert(model->isTrained());

    std::vector<float> output;
    int result = (int) model->predict(img, output);
    return {result, output[result]};
}

int MLPHand::exportModelTo(const std::string xmlFileName) {
    assert(model->isTrained());

    if (!xmlFileName.empty()) {
        LOGP_I(this, "Exporting model to " + xmlFileName);
        model->save(xmlFileName);
        LOGP_I(this, "Model successfully exported");
        return Code::SUCCESS;
    }

    LOGP_E(this, "ERROR: model not exported");
    return Code::ERROR;
}

std::pair<double, std::map<int, StatPredict *>> MLPHand::testOn(const cv::Mat &testData, const cv::Mat &testResponses) {
    assert(model->isTrained());

    std::map<int, StatPredict *> statMap;

    double nbOfSamples = testData.rows;
    double totalSuccess = 0.0;

    // Compute prediction error on test data
    // We count the number of prediction success
    for (int i = 0; i < nbOfSamples; ++i) {
        // Get response
        int response = testResponses.at<int>(i);

        // Predict output
        std::vector<float> predictOutput;
        int prediction = (int) model->predict(testData.row(i), predictOutput);
        float trustPercentage = predictOutput[prediction];

        // Check if already in the stat map
        if (statMap.find(response) == statMap.end()) {
            statMap[response] = new StatPredict(response);
        }

        // Add computed data to whatever we are calculating
        bool success = std::abs(BASE_LETTER + prediction - response) <= FLT_EPSILON;
        totalSuccess += success ? 1 : 0;

        statMap[response]->pushStat(success, BASE_LETTER + prediction, trustPercentage, predictOutput);
    }

    double successRate = totalSuccess / nbOfSamples;

    return {successRate, statMap};
}