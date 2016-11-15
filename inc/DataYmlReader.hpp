//
// @author Loris Friedel
//

#pragma once

#include <cv.hpp>
#include <string>

class DataYmlReader {
public:
    DataYmlReader(std::string filePath);

    /**
     * Read an yml file containing data for training.
     *
     * @param data_output Read data from file. Must be an array.
     * @param letter_output Read letter from file. Must be the ASCII code.
     * @return true if reading succeed, false otherwise.
     */
    int read(cv::Mat &data_output, int &letter_output);

private:
    std::string filePath;
};

