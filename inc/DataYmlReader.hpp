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
     * @param dataOutput Read data from file. Must be an array.
     * @param labelOutput Read label from file. Must be the an integer.
     * @return true if reading succeed, false otherwise.
     */
    int read(cv::Mat &dataOutput, int &labelOutput);

private:
    std::string filePath;
};

