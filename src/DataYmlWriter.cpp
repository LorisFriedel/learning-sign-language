//
// @author Loris Friedel
//

#include "../inc/DataYmlWriter.hpp"
#include "../inc/constant.h"
#include "../inc/log.h"

DataYmlWriter::DataYmlWriter(std::string filePath)
        : filePath(filePath) {}

int DataYmlWriter::write(cv::Mat &dataInput, int labelInput) {
    cv::FileStorage fs(filePath, cv::FileStorage::WRITE);

    if (fs.isOpened()) {
        fs << Default::KEY_LABEL << labelInput;
        fs << Default::KEY_DATA << dataInput;
        fs.release();
        return true;
    } else {
        LOG_E("ERROR: Could not write data");
        return false;
    }
}

int DataYmlWriter::writeLetter(cv::Mat &dataInput, int letterInput) {
    cv::FileStorage fs(filePath, cv::FileStorage::WRITE);

    if (fs.isOpened()) {
        fs << Default::KEY_LETTER << letterInput;
        fs << Default::KEY_MAT << dataInput;
        fs.release();
        return true;
    } else {
        LOG_E("ERROR: Could not write data");
        return false;
    }
}

int DataYmlWriter::write(std::vector<float> &dataInput, int labelInput) {
    cv::Mat matData(cv::Size((int) dataInput.size(), 1), CV_32FC1);
    for(int i = 0; i < dataInput.size(); i++) {
        matData.at<float>(i) = dataInput[i];
    }
    return write(matData, labelInput);
}