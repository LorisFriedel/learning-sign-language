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
        // Get label
        if(!fs[Default::KEY_LETTER].empty()) {
            fs[Default::KEY_LETTER] >> labelOutput;
            labelOutput -= 'a';
        } else if(!fs[Default::KEY_LABEL].empty()) {
            fs[Default::KEY_LABEL] >> labelOutput;
        } else {
            LOG_E("ERROR: Cant find label in yml file");
            return false;
        }

        // Get data
        if(!fs[Default::KEY_MAT].empty()) {
            fs[Default::KEY_MAT] >> dataOutput;
        } else if(!fs[Default::KEY_DATA].empty()) {
            fs[Default::KEY_DATA] >> dataOutput;
        } else {
            LOG_E("ERROR: Cant find data in yml file");
            return false;
        }

        fs.release();
        return true;
    } else {
        LOG_E("ERROR: Could not read data");
        return false;
    }
}
