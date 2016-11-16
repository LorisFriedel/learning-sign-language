//
//  @author Loris Friedel
//

#include "../inc/HandTracker.hpp"
#include "../inc/log.h"
#include "../inc/CamshiftRunner.hpp"

#define MARGIN_HIDE 32

HandTracker::HandTracker() {}

cv::RotatedRect HandTracker::trackHand(const cv::Mat &img, const cv::Mat backproj, const cv::Rect faceRect) {
    if (imgRect.size() != img.size()) {
        imgRect = cv::Rect(0, 0, img.cols, img.rows);
    }

    // If we found the face, we can detect the hand
    if (faceRect.area() > 1) {
        // Create mask to hide the face in the backproj
        cv::Mat mask(img.rows, img.cols, CV_8UC1, cv::Scalar(255));
        const cv::Rect toHide(faceRect.x - MARGIN_HIDE, 0, faceRect.width + MARGIN_HIDE, mask.cols);
        mask(toHide & imgRect) = 0;

        // Apply mask to the old backproj
        cv::Mat newBackproj(backproj & mask);

        // Find hand using camshift and the new backproj
        return cv::CamShift(newBackproj, imgRect,
                            cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));
    } else {
        // Face has been lost
        LOG_I("Face lost in hand tracker");
        return Geo::NULL_ROTATEDRECT;
    }
}
