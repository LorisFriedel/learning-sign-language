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
     * @param dataInput Write data into the file. Must be an array.
     * @param labelInput Write the letter into the file. Must be the ASCII code.
     * @return true if writing succeed, false otherwise.
     */
    int write(cv::Mat &dataInput, int labelInput);

    int write(std::vector<float> &dataInput, int labelInput);

private:
    std::string filePath;
};

