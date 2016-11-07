//
//  @author Loris Friedel
//

#pragma once

#include <cv.hpp>
#include <functional>
#include "CamshiftTracker.hpp"

class VideoStreamReader;
class ObjectDetector;

class CamshiftRunner {
public:
    CamshiftRunner(VideoStreamReader &vsr, ObjectDetector &object_detector);

    int run_tracking(const std::function<void(CamshiftTracker &, const cv::Mat &,
                                              const cv::RotatedRect &, const bool)> *success_callback = nullptr);

    void stop();

    void recalibrate();

    CamshiftTracker c_tracker;

private:
    VideoStreamReader &_vsr;
    ObjectDetector &_object_detector;

    // Control variables
    bool _stop;
    bool _recalibrate;
};

