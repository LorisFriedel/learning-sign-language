//
//  @author Loris Friedel
//

#pragma once

#include <cv.hpp>

class HandTracker {

public:
    /**
     * Instantiate a hand tracker.
     * @return
     */
    HandTracker();

    /**
     * Track the hand on the given image
     * @param t_img Image on which to perform the tracking (will not be modified)
     * @return A rotated rectangle that reprents the best new position found of the tracked hand
     */
    cv::RotatedRect track_hand(const cv::Mat &img, const cv::Mat backproj, const cv::Rect face_rect);

private:
    cv::Rect _img_rect;
};

