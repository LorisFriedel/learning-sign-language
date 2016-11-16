//
// @author Loris Friedel
//

#include "../inc/DataYmlReader.hpp"
#include "../inc/log.h"
#include "../inc/constant.h"

DataYmlReader::DataYmlReader(std::string filePath) : filePath(filePath) {}

int DataYmlReader::read(cv::Mat &dataOutput, int &letterOutput) {
    cv::FileStorage fs(filePath, cv::FileStorage::READ);

    if (fs.isOpened()) {
        fs[Default::KEY_LETTER] >> letterOutput;
        fs[Default::KEY_MAT] >> dataOutput;
        fs.release();
        return true;
    } else {
        LOG_E("ERROR: Could not read data");
        return false;
    }
}
