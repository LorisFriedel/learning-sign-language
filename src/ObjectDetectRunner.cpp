//
//  @author Loris Friedel
//

#include "../inc/ObjectDetectRunner.hpp"
#include "../inc/VideoStreamReader.hpp"
#include "../inc/code.h"
#include "../inc/log.h"
#include "../inc/colors.h"

ObjectDetectRunner::ObjectDetectRunner(VideoStreamReader &vsr, const cv::CascadeClassifier &cascade)
        : _vsr(vsr), _object_detector(cascade) {}

int ObjectDetectRunner::run_detection() {
    cv::namedWindow("webcam_window");
    std::function<void(cv::Mat, bool, double, cv::Rect)> F_handle_result =
            [this](cv::Mat img, bool detected,
                                    double timeMs, cv::Rect roi) {
                LOG_I("Detection performed in " << timeMs << "ms");
                // Refresh webcam image with drawings if needed
                if (!img.empty()) {
                    // Draw result on image
                    this->drawResult(img, roi, detected);
                    cv::imshow("webcam_window", img);
                }

                // Wait for windows update and maybe catch a quit action
                int c = cv::waitKey(1);
                if (c == 27 || c == 'q' || c == 'Q') {
                    this->_object_detector.stop_loop_detect();
                }
            };

    _object_detector.loop_detect(_vsr, F_handle_result);

    _vsr.closeStream();

    return Code::SUCCESS;
}

void ObjectDetectRunner::drawResult(cv::Mat &img, cv::Rect &roi, const bool detected) {
    if (detected) {
        // Display green dot to say "face detected!"
        cv::circle(img, cv::Point(16, 16), 8, Color::GREEN, cv::FILLED, cv::FILLED, 0);
    }

    ///// Circle (or rectangle) drawing /////
    cv::Point center;
    int radius;

    // Draw regarding ratio (almost useless)
    center.x = cvRound((roi.x + roi.width * 0.5));
    center.y = cvRound((roi.y + roi.height * 0.5));
    radius = cvRound((roi.width + roi.height) * 0.25);
    circle(img, center, radius, Color::RED, 1);
}