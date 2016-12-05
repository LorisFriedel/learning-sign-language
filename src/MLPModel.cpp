//
//  @author Loris Friedel
//

#include <chrono>
#include <iterator>
#include <fstream>
#include "../inc/MLPModel.hpp"
#include "../inc/log.h"
#include "../inc/code.h"
#include "../inc/Timer.hpp"

MLPModel::MLPModel(int nbOfHiddenLayer, int nbOfNeuron) {
    for (int i = 0; i < nbOfHiddenLayer; i++) {
        hiddenLayers.push_back(nbOfNeuron);
    }
}

MLPModel::MLPModel(const std::vector<int> topologyPattern)
        : hiddenLayers(topologyPattern) {}

MLPModel::MLPModel(const std::string topologyPattern) {
    std::stringstream ss(topologyPattern);
    std::string tok;

    char delimiter = topologyPattern.find('_') != std::string::npos ? '_' : ' ';

    while (getline(ss, tok, delimiter)) {
        hiddenLayers.push_back(std::stoi(tok));
    }
}

void MLPModel::setMaxIter(int maxIteration) {
    this->maxIter = maxIteration;
}

void MLPModel::setMethod(cv::ml::ANN_MLP::TrainingMethods method) {
    this->method = method;
}

void MLPModel::setMethodEpsilon(double epsilon) {
    this->methodEpsilon = epsilon;
}

void MLPModel::setLabelMap(LabelMap labelMap) {
    this->labelMap = labelMap;
}

void MLPModel::exportTrainDataDistribution(const std::string jsonFilePath) {
    jsonDistribFilePath = jsonFilePath;

    if (!jsonFilePath.empty()) {
        LOGP_I(this, "Exporting training data distribution...");

        std::fstream jsonFile;
        jsonFile.open(jsonFilePath, std::fstream::in | std::fstream::out | std::fstream::app);

        // If file does not exist, create new file
        if (!jsonFile) {
            jsonFile.open(jsonFilePath, std::fstream::in | std::fstream::out | std::fstream::trunc);
        }

        auto finalIter = classesCountMap.end();
        finalIter--;
        jsonFile << "{\n";
        for (auto it = classesCountMap.begin(); it != finalIter; ++it) {
            std::string key = labelMap.get(it->first);
            jsonFile << "\"" << key << "\" : " << it->second << ",\n";
        }
        jsonFile << "\"" << std::to_string(finalIter->first) << "\" : "
                 << finalIter->second << "\n";
        jsonFile << "}";

        jsonFile.close();

        LOGP_I(this, "Training data distribution successfully exported to " << jsonFilePath);
    }
}

int MLPModel::learnFrom(const std::string classifier_file_name) {
    model = cv::ml::StatModel::load<cv::ml::ANN_MLP>(classifier_file_name);

    LOGP_I(this, "Loading classifier...");
    if (model.empty()) {
        LOGP_E(this, "ERROR: Could not read the classifier : " << classifier_file_name);
        return Code::CASCADE_LOAD_ERROR;
    } else {
        cv::Mat layers = model.get()->getLayerSizes();
        hiddenLayers.clear();
        inputSize = model.get()->getLayerSizes().row(0).at<int>(0);
        outputSize = model.get()->getLayerSizes().row(layers.rows - 1).at<int>(0);

        for (int i = 1; i < layers.rows - 1; i++) {
            int layerSize = model.get()->getLayerSizes().row(i).at<int>(0);
            hiddenLayers.push_back(layerSize);
        }

        LOGP_I(this, "Classifier " << classifier_file_name << " successfully loaded!");
        return Code::SUCCESS;
    }
}

inline cv::TermCriteria MLPModel::TC(int iters, double eps) {
    return cv::TermCriteria(cv::TermCriteria::MAX_ITER + (eps > 0 ? cv::TermCriteria::EPS : 0), iters, eps);
}

int MLPModel::learnFrom(const cv::Mat &trainingData, const cv::Mat &trainingResponses) {
    Timer timeMonitor;

    // Start timer
    timeMonitor.start();

    const int nbOutputClasses = trainingData.cols;
    int nbOfSamples = trainingData.rows;

    cv::Mat formattedResponses = cv::Mat::zeros(nbOfSamples, nbOutputClasses, CV_32FC1);

    // Unrolling the responses
    LOGP_I(this, "Formatting responses...");
    classesCountMap.clear();
    for (int i = 0; i < nbOfSamples; i++) {
        int response = trainingResponses.at<int>(i);
        formattedResponses.at<float>(i, response) = 1.f;

        if (classesCountMap.find(response) == classesCountMap.end()) {
            classesCountMap[response] = 0;
        }
        classesCountMap[response]++;
    }
    LOGP_I(this, "Formatting responses done!");

    if (!jsonDistribFilePath.empty()) {
        exportTrainDataDistribution(jsonDistribFilePath);
    }

    LOGP_I(this, "Training samples composition: ");
    for (auto it = classesCountMap.begin(); it != classesCountMap.end(); ++it) {
        std::string key = labelMap.get(it->first);
        LOGP_I(this, " - " << key << " * " << it->second);
    }
    LOGP_I(this, "");

    // Create and configure layers
    std::vector<int> layerSizes;
    layerSizes.push_back(trainingData.cols);
    for (int i = 0; i < hiddenLayers.size(); i++) {
        layerSizes.push_back(hiddenLayers[i]);
    }
    layerSizes.push_back(nbOutputClasses);

    // Train classifier
    cv::Ptr<cv::ml::TrainData> tData =
            cv::ml::TrainData::create(trainingData, cv::ml::ROW_SAMPLE, formattedResponses);

    LOGP_I(this, "Training the classifier (" << nbOfSamples << " samples) - layer pattern: " << getTopologyStr()
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

std::pair<int, float> MLPModel::predict(cv::Mat &input) {
    assert(model->isTrained());

    std::vector<float> output;
    int result = (int) model->predict(input, output);
    return {result, output[result]};
}

int MLPModel::exportModelTo(const std::string xmlFileName) {
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

std::pair<double, std::map<int, StatPredict *>>
MLPModel::testOn(const cv::Mat &testData, const cv::Mat &testResponses) {
    assert(model->isTrained());

    std::map<int, StatPredict *> statMap;

    int nbOfSamples = testData.rows;
    int totalSuccess = 0;

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
        bool success = std::abs(prediction - response) <= FLT_EPSILON;
        totalSuccess += success ? 1 : 0;

        statMap[response]->pushStat(success, prediction, trustPercentage, predictOutput);
    }

    double successRate = (double) totalSuccess / (double) nbOfSamples;

    return {successRate, statMap};
}

std::string MLPModel::getTopologyStr() {
    std::stringstream patternStream;

    patternStream << "[";
    patternStream << inputSize << ":";
    for (int i = 0; i < hiddenLayers.size(); i++) {
        patternStream << hiddenLayers[i] << ":";
    }
    patternStream << outputSize << "]";

    return patternStream.str();
}

std::string MLPModel::convertLabel(int label) {
    return labelMap.get(label);
}

const LabelMap &MLPModel::getLabelMap() const {
    return labelMap;
}
