//
//  @author Loris Friedel
//

#pragma once

#include <opencv2/core/mat.hpp>
#include <ml.h>

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
     * @param test_data Data to test.
     * @param test_responses Responses for the data set.
     * @return The average of success between [0, 1]. 0 mean no prediction success, 1 mean no prediction error.
     */
    double testOn(const cv::Mat &test_data, const cv::Mat &test_responses);

    /**
     * Export the current model to a file.
     *
     * @param xml_file_name Path to the file where to export the data model as xml.
     * @return success code
     */
    int exportModelTo(const std::string xml_file_name);

private:
    std::vector<int> networkPattern;
    cv::Ptr<cv::ml::ANN_MLP> model;

    inline cv::TermCriteria TC(int iters, double eps);
};

