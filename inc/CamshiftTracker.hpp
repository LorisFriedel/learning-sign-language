//
// @author Loris Friedel
//

#pragma once

#include "geo.h"
#include <opencv2/bgsegm.hpp>

class CamshiftTracker {

public:
    /**
     * Calibrate the object tracker using given parameters
     * @param initial_img Image to use for calibration (will not be modified)
     * @param initial_rect_obj Rectangle where the object is on the given image to use for calibration
     * @return
     */
    CamshiftTracker(const cv::Mat &initial_img, const cv::Rect &initial_rect_obj);

    /**
     * Initialize a camshift trachker without calibration.
     * @return 
     */
    CamshiftTracker();

    /**
     * Recalibrate the tracking using a new image and a new object rectangle
     * @param img Image to use for calibration (will not be modified)
     * @param rect_obj Rectangle of the object on the given image to use for calibration
     */
    void recalibrate(const cv::Mat &img, const cv::Rect &rect_obj);

    /**
     * Track the object on the given image
     * @param t_img Image on which to perform the tracking (will not be modified)
     * @return A rotated rectangle that reprents the best new position found of the tracked object
     */
    cv::RotatedRect track_obj(const cv::Mat &img);

    /**
     * @return the last generated backprof of the tracking method
     */
    const cv::Mat &get_backproj() const;

    int v_min = 10;
    int v_max = 256;
    int s_min = 32;
    int threshold = 160;
    int hsize = 16;

private:
    // Private methods and functions
    void compute_colors(const cv::Mat &img);

    // Private fields
    cv::Rect track_window;
    cv::Mat hsv, hue, mask, hist, backproj;
};
