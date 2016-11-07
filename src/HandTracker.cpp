//
//  @author Loris Friedel
//

#include "../inc/HandTracker.hpp"
#include "../inc/log.h"
#include "../inc/CamshiftRunner.hpp"

#define MARGIN_HIDE 32

HandTracker::HandTracker() {}

cv::RotatedRect HandTracker::track_hand(const cv::Mat &img, const cv::Mat backproj, const cv::Rect face_rect) {
    if (_img_rect.size() != img.size()) {
        _img_rect = cv::Rect(0, 0, img.cols, img.rows);
    }

    // If we found the face, we can detect the hand
    if (face_rect.area() > 1) {
        // Create mask to hide the face in the backproj
        cv::Mat mask(img.rows, img.cols, CV_8UC1, cv::Scalar(255));
        const cv::Rect to_hide(face_rect.x - MARGIN_HIDE, 0, face_rect.width + MARGIN_HIDE, mask.cols);
        mask(to_hide & _img_rect) = 0;

        // Apply mask to the old backproj
        cv::Mat new_backproj(backproj & mask);

        // Find hand using camshift and the new backproj
        return cv::CamShift(new_backproj, _img_rect,
                            cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1));
    } else {
        // Face has been lost
        LOG_I("Face lost in hand tracker");
        return Geo::NULL_ROTATEDRECT;
    }
}
