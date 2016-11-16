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
    cv::RotatedRect trackHand(const cv::Mat &img, const cv::Mat backproj, const cv::Rect faceRect);

private:
    cv::Rect imgRect;
};

