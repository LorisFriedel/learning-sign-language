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

CamshiftTracker::CamshiftTracker(const cv::Mat &initial_img, const cv::Rect &initial_rect_obj) {
    recalibrate(initial_img, initial_rect_obj);
}

CamshiftTracker::CamshiftTracker() {}

void CamshiftTracker::compute_colors(const cv::Mat &img) {
    cv::cvtColor(img, hsv, CV_BGR2HSV);

    cv::inRange(hsv, cv::Scalar(0, s_min, MIN(v_min, v_max)),
                cv::Scalar(180, 256, MAX(v_min, v_max)), mask);
    int ch[] = {0, 0};
    hue.create(hsv.size(), hsv.depth());
    cv::mixChannels(&hsv, 1, &hue, 1, ch, 1);
}

void CamshiftTracker::recalibrate(const cv::Mat &img, const cv::Rect &rect_obj) {
    compute_colors(img);

    float hranges[] = {0, 180};
    const float *phranges = hranges;
    cv::Mat roi(hue, rect_obj), maskroi(mask, rect_obj);
    cv::calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
    cv::normalize(hist, hist, 0, 255, CV_MINMAX);

    track_window = rect_obj;

    // TODO run auto calibration for vmin vmax smin and threshold
}

cv::RotatedRect CamshiftTracker::track_obj(const cv::Mat &t_img) {
    cv::RotatedRect track_box;

    compute_colors(t_img);

    float hranges[] = {0, 180};
    const float *phranges = hranges;
    cv::calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
    backproj &= mask;
    cv::threshold(backproj, backproj, threshold, 256, CV_THRESH_BINARY);

    if (track_window.area() <= 1) {
        LOG_E("Face tracker has lost face to track");
        return Geo::NULL_ROTATEDRECT;
    } else {
        track_box = cv::CamShift(backproj, track_window,
                                 cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));
        return track_box;
    }
}

const cv::Mat &CamshiftTracker::get_backproj() const {
    return backproj;
}
