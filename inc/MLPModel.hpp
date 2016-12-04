//
//  @author Loris Friedel
//

#pragma once

#include <opencv2/core/mat.hpp>
#include <ml.h>
#include "StatPredict.hpp"

class MLPModel {

public:
    /**
     * Instantiate a MLP model.
     *
     * @param nbOfHiddenLayer Number of hidden layer for the neural network.
     * @param nbOfNeuron Number of neuron per layer.
     * @return
     */
    MLPModel(int nbOfHiddenLayer = 0, int nbOfNeuron = 0);

    /**
     * Instantiate a MLP model.
     *
     * @param topologyPattern Pattern for layer and neuron configuration
     * @return
     */
    MLPModel(const std::vector<int> topologyPattern);

    /**
     * Instantiate a MLP model.
     *
     * @param topologyPattern Pattern (string representation) for layer and neuron configuration (e.g. "8 32 16")
     * @return
     */
    MLPModel(const std::string topologyPattern);

    /**
     * TODO
     * @param maxIteration
     */
    void setMaxIter(int maxIteration);

    /**
     * TODO
     * @param method
     */
    void setMethod(cv::ml::ANN_MLP::TrainingMethods method);

    /**
     * TODO
     * @param epsilon
     */
    void setMethodEpsilon(double epsilon);

    /**
     *
     * @param labelMap
     */
    void setLabelMapping(std::map<int, std::string> labelMap);

    /**
     * Export the training data distribution to the specified json file.
     * If the model is already trained, exportation is made when the method is called.
     * If not, the exportation is made during the training.
     *
     * If you really want to not export log when training the model a second time, you can disable this
     * functionality by passing an empty string to this method.
     *
     * @param jsonFilePath Path to a json file (already existing or to be created)
     */
    void exportTrainDataDistribution(const std::string jsonFilePath);

    /**
     * Teach the model from an existing classifier.
     *
     * @param classifier_file_name Path to the classifier file.
     * @return true if reading succeed, false otherwise.
     */
    int learnFrom(const std::string classifier_file_name);

    /**
     * Teach the model from a data set.
     *
     * @param trainingData Data to use for training.
     * @param trainingResponses Responses for the data set.
     * @return true if reading succeed, false otherwise.
     */
    int learnFrom(const cv::Mat &trainingData, const cv::Mat &trainingResponses);

    /**
     * Use the current model to predict a result using the given data.
     *
     * @param input Data to use for prediction
     * @return A pair: <0> integer (representing a letter from 0 to 26), <1> the probability to be right
     */
    std::pair<int, float> predict(cv::Mat &input);

    /**
     * Test the given data set on the current model.
     *
     * @param testData Data to test.
     * @param testResponses Responses for the data set.
     * @return The average of success between [0, 1]. 0 mean no prediction success, 1 mean no prediction error,
     * plus a map with details about the test and prediction
     */
    std::pair<double, std::map<int, StatPredict *>> testOn(const cv::Mat &testData, const cv::Mat &testResponses);

    /**
     * Export the current model to a file.
     *
     * @param xmlFileName Path to the file where to export the data model as xml.
     * @return success code
     */
    int exportModelTo(const std::string xmlFileName);

    /**
     * @return the topology of this model in string format
     */
    std::string getTopologyStr();

    /**
     * Return the string representation of the given label
     * @param label Label used in this model
     * @return a string representing this label
     */
    std::string convertLabel(int label);

private:
    std::vector<int> hiddenLayers;
    int inputSize;
    int outputSize;

    cv::Ptr<cv::ml::ANN_MLP> model;

    std::map<int, int> classesCountMap;
    std::string jsonDistribFilePath;
    std::map<int, std::string> labelMap;

    int method = cv::ml::ANN_MLP::BACKPROP;
    double methodEpsilon = 0.001;
    int maxIter = 128;

    inline cv::TermCriteria TC(int iters, double eps);
};

