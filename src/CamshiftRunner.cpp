//
//  @author Loris Friedel
//
#include "../inc/CamshiftRunner.hpp"

#include "../inc/VideoStreamReader.hpp"
#include "../inc/KeyInputHandler.hpp"
#include "../inc/ObjectDetector.hpp"
#include "../inc/code.h"
#include "../inc/log.h"

CamshiftRunner::CamshiftRunner(VideoStreamReader &vsr, ObjectDetector &object_detector)
        : _vsr(vsr), _object_detector(object_detector), c_tracker(),
          _stop(false), _recalibrate(false) {}

int CamshiftRunner::run_tracking(
        const std::function<void(CamshiftTracker &, const cv::Mat &, const cv::RotatedRect &,
                                 const bool)> *success_callback) {
    // Init necessary stuff to detect the object from scratch
    cv::Mat img;
    cv::Rect detected_object;
    bool object_found;

    // Detect the object for camshift calibration
    std::tuple<bool, double, cv::Rect> detection_result;
    bool face_detected = false;
    while (!face_detected) {
        img = _vsr.readFrame();
        detection_result = _object_detector.detect_in(img);
        if (std::get<0>(detection_result)) {
            face_detected = true;
            detected_object = std::get<2>(detection_result);
        }
    }

    // Create the camshift tracker and init it with the previously detected object
    c_tracker.recalibrate(img, detected_object);

    // Current result for face tracking
    cv::RotatedRect object_tracked;

    while (!_stop) {
        // Read a frame
        img = _vsr.readFrame();

        // Recalibrate if needed
        if (_recalibrate) {
            LOG_I("Recalibrating tracker...");
            detection_result = _object_detector.detect_in(img);
            c_tracker.recalibrate(img, std::get<2>(detection_result));
            _recalibrate = !_recalibrate;
        }

        // Tracking
        object_tracked = c_tracker.track_obj(img);
        object_found = object_tracked.boundingRect().area() > 1;

        // Tracked object is detected : we execute the callback
        if (success_callback != nullptr) {
            (*success_callback)(c_tracker, img, object_tracked, object_found);
        }

        // Check for recalibration
        if (!object_found) {
            _recalibrate = true;
        }
    }

    _vsr.closeStream();
    return Code::SUCCESS;
}

void CamshiftRunner::stop() {
    _stop = true;
    LOG_I("Stopping camshift runner..");
}

void CamshiftRunner::recalibrate() {
    _recalibrate = true;
}
