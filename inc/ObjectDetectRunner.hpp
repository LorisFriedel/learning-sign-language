//
//  @author Loris Friedel
//

#pragma once

#include "ObjectDetector.hpp"

class VideoStreamReader;

class ObjectDetectRunner {
public:
    ObjectDetectRunner(VideoStreamReader &vsr, const cv::CascadeClassifier &cascade);

    int runDetection();

private:
    ObjectDetector objectDetector;
    VideoStreamReader &vsr;

    void drawResult(cv::Mat &img, cv::Rect &roi, const bool detected);
};

