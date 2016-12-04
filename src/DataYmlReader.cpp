//
// @author Loris Friedel
//

#include "../inc/DataYmlReader.hpp"
#include "../inc/log.h"
#include "../inc/constant.h"

DataYmlReader::DataYmlReader(std::string filePath) : filePath(filePath) {}

int DataYmlReader::read(cv::Mat &dataOutput, int &labelOutput) {
    cv::FileStorage fs(filePath, cv::FileStorage::READ);

    if (fs.isOpened()) {
        fs[Default::KEY_LETTER] >> labelOutput;
        // TODO remove the next line to have a generalized learning program
        labelOutput -= 'a'; // TODO this is "sign language" specific, need to be generalized.
        fs[Default::KEY_MAT] >> dataOutput;
        fs.release();
        return true;
    } else {
        LOG_E("ERROR: Could not read data");
        return false;
    }
}
