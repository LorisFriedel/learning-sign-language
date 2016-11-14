//
// @author Loris Friedel
//

#include <cv.hpp>
#include "../inc/DataYmlReader.hpp"
#include "../inc/log.h"
#include "../inc/constant.h"

DataYmlReader::DataYmlReader(std::string filePath) : filePath(filePath) {}

int DataYmlReader::read(cv::Mat &data_output, int &letter_output) {
    cv::FileStorage fs(filePath, cv::FileStorage::READ);

    if (fs.isOpened()) {
        fs[Default::KEY_LETTER] >> letter_output;
        fs[Default::KEY_MAT] >> data_output;
        fs.release();
        return true;
    } else {
        LOG_E("ERROR: Could not read data");
        return false;
    }
}
