//
// @author Loris Friedel
//

#pragma once

#include <string>
#include <cv.hpp>

class DataYmlWriter {
public:
    DataYmlWriter(std::string filePath);

    /**
     * Read an yml file containing data for training.
     *
     * @param data_input Write data into the file. Must be an array.
     * @param letter_input Write the letter into the file. Must be the ASCII code.
     * @return true if writing succeed, false otherwise.
     */
    int write(cv::Mat &data_input, int letter_input);

    int write(std::vector<float> &data_input, int letter_input);

private:
    std::string filePath;
};

