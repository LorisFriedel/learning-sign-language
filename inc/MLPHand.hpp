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
     * @param nb_of_hidden_layer Number of hidden layer for the neural network.
     * @param nb_of_neuron Number of neuron per layer.
     * @return
     */
    MLPHand(int nb_of_hidden_layer = 0, int nb_of_neuron = 0);

    /**
     * Teach the model from an existing classifier.
     *
     * @param classifier_file_name Path to the classifier file.
     * @return true if reading succeed, false otherwise.
     */
    int learn_from(const std::string classifier_file_name);

    /**
     * Teach the model from a data set.
     *
     * @param training_data Data to use for training.
     * @param training_responses Responses for the data set.
     * @return true if reading succeed, false otherwise.
     */
    int learn_from(const cv::Mat &training_data, const cv::Mat &training_responses);

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
    double test_on(const cv::Mat &test_data, const cv::Mat &test_responses);

    /**
     * Export the current model to a file.
     *
     * @param xml_file_name Path to the file where to export the data model as xml.
     */
    void export_model_to(const std::string xml_file_name);

private:
    int _nb_of_hidden_layer;
    int _nb_of_neuron;
    cv::Ptr<cv::ml::ANN_MLP> _model;

    inline cv::TermCriteria TC(int iters, double eps);
};

