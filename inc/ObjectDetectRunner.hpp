//
//  @author Loris Friedel
//

#pragma once

#include "ObjectDetector.hpp"

class VideoStreamReader;

class ObjectDetectRunner {
public:
    ObjectDetectRunner(VideoStreamReader &vsr, const cv::CascadeClassifier &cascade);

    int run_detection();

private:
    ObjectDetector _object_detector;
    VideoStreamReader &_vsr;

    void drawResult(cv::Mat &img, cv::Rect &roi, const bool detected);
};

