//
// @author Loris Friedel
//

#include "../inc/DataYmlWriter.hpp"
#include "../inc/constant.h"
#include "../inc/log.h"

DataYmlWriter::DataYmlWriter(std::string filePath)
        : filePath(filePath) {}

int DataYmlWriter::write(cv::Mat &data_input, int letter_input) {
    cv::FileStorage fs(filePath, cv::FileStorage::WRITE);

    if (fs.isOpened()) {
        fs << Default::KEY_LETTER << letter_input;
        fs << Default::KEY_MAT << data_input;
        fs.release();
        return true;
    } else {
        LOG_E("ERROR: Could not write data");
        return false;
    }
}

int DataYmlWriter::write(std::vector<float> &data_input, int letter_input) {
    cv::Mat mat_data(cv::Size((int) data_input.size(), 1), CV_32FC1);
    for(int i = 0; i < data_input.size(); i++) {
        mat_data.at<float>(i) = data_input[i];
    }
    return write(mat_data, letter_input);
}