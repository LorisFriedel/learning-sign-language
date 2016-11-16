//
// Created by loris on 9/19/16.
//

#include "../inc/VideoStreamReader.hpp"

#include "opencv2/objdetect.hpp"
#include "../inc/log.h"
#include "../inc/geo.h"
#include "../inc/code.h"

using namespace cv;

int VideoStreamReader::openStream(const int input) {
    if (capture.open(input)) {
        LOG_I("Capture from camera #" << input << " successful");
        return Code::SUCCESS;
    } else {
        LOG_E("Capture from camera #" << input << " didn't work");
        return Code::ERROR;
    }
}

int VideoStreamReader::openStream(const std::string filename) {
    if (capture.open(filename)) {
        LOG_I("Capture from camera #" << filename << " successful");
        return Code::SUCCESS;
    } else {
        LOG_E("Capture from camera #" << filename << " didn't work");
        return Code::ERROR;
    }
}

Mat VideoStreamReader::readFrame() {
    if (capture.isOpened()) {
        capture >> frame;
        return frame;
    } else {
        return Geo::NULL_MAT;
    }
}

void VideoStreamReader::closeStream() {
    capture.release();
    LOG_I("Capture stopped");
}