//
// Created by loris on 9/19/16.
//

#pragma once

#include <opencv2/videoio.hpp>

class VideoStreamReader {

public:
    int openStream(const int input);
    int openStream(const std::string filename);

    cv::Mat readFrame();

    void closeStream();

private:
    cv::VideoCapture capture;
    cv::Mat frame;
};