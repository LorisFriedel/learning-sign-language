//
//  @author Loris Friedel
//

#pragma once

#include <opencv2/core/mat.hpp>
#include <ml.h>
#include "StatPredict.hpp"

class MLPHand {

public:
    /**
     * Instantiate a MLP hand learner.
     *
     * @param nbOfHiddenLayer Number of hidden layer for the neural network.
     * @param nbOfNeuron Number of neuron per layer.
     * @return
     */
    MLPHand(int nbOfHiddenLayer = 0, int nbOfNeuron = 0);

    /**
     * Instantiate a MLP hand learner.
     *
     * @param networkPattern Pattern for layer and neuron configuration
     * @return
     */
    MLPHand(const std::vector<int> networkPattern);

    /**
     * Instantiate a MLP hand learner.
     *
     * @param networkPattern Pattern (string representation) for layer and neuron configuration (e.g. "8 32 8")
     * @return
     */
    MLPHand(const std::string networkPatternStr);

    // TODO setOutputJson(File *jsonFile);
    // TODO add an out file attribute to be used instead (or in parralel) of logging data in a json or yml file ?

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
     * @param img Data to use for prediction
     * @return A pair: <0> integer (representing a letter from 0 to 26), <1> the probability to be right
     */
    std::pair<int, float> predict(cv::Mat &img);

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

private:
    std::vector<int> networkPattern;
    cv::Ptr<cv::ml::ANN_MLP> model;

    inline cv::TermCriteria TC(int iters, double eps);
};

