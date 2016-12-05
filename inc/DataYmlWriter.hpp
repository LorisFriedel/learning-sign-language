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
     * Write in an yml file given data and label.
     *
     * @param dataInput Data to be written into the file.
     * @param labelInput Label to be written into the file.
     * @return true if writing succeed, false otherwise.
     */
    int write(cv::Mat &dataInput, int labelInput);

    int writeLetter(cv::Mat &dataInput, int letterInput);

    int write(std::vector<float> &dataInput, int labelInput);

private:
    std::string filePath;
};

