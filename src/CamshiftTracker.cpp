//
// @author Loris Friedel
//

#include <opencv2/core/base.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <cv.hpp>
#include "../inc/CamshiftTracker.hpp"
#include "../inc/log.h"
#include "../inc/geo.h"

CamshiftTracker::CamshiftTracker(const cv::Mat &initialImg, const cv::Rect &initialRectObj) {
    recalibrate(initialImg, initialRectObj);
}

CamshiftTracker::CamshiftTracker() {}

void CamshiftTracker::computeColors(const cv::Mat &img) {
    cv::cvtColor(img, hsv, CV_BGR2HSV);

    cv::inRange(hsv, cv::Scalar(0, sMin, MIN(vMin, vMax)),
                cv::Scalar(180, 256, MAX(vMin, vMax)), mask);
    int ch[] = {0, 0};
    hue.create(hsv.size(), hsv.depth());
    cv::mixChannels(&hsv, 1, &hue, 1, ch, 1);
}

void CamshiftTracker::recalibrate(const cv::Mat &img, const cv::Rect &rectObj) {
    computeColors(img);

    float hranges[] = {0, 180};
    const float *phranges = hranges;
    cv::Mat roi(hue, rectObj), maskroi(mask, rectObj);
    cv::calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
    cv::normalize(hist, hist, 0, 255, CV_MINMAX);

    trackWindow = rectObj;

    // TODO run auto calibration for vmin vmax smin and threshold
}

cv::RotatedRect CamshiftTracker::trackObj(const cv::Mat &t_img) {
    cv::RotatedRect track_box;

    computeColors(t_img);

    float hranges[] = {0, 180};
    const float *phranges = hranges;
    cv::calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
    backproj &= mask;
    cv::threshold(backproj, backproj, threshold, 256, CV_THRESH_BINARY);

    if (trackWindow.area() <= 1) {
        LOG_E("Face tracker has lost face to track");
        return Geo::NULL_ROTATEDRECT;
    } else {
        track_box = cv::CamShift(backproj, trackWindow,
                                 cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));
        return track_box;
    }
}

const cv::Mat &CamshiftTracker::getBackproj() const {
    return backproj;
}
