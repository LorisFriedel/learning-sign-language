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
     * @param initialImg Image to use for calibration (will not be modified)
     * @param initialRectObj Rectangle where the object is on the given image to use for calibration
     * @return
     */
    CamshiftTracker(const cv::Mat &initialImg, const cv::Rect &initialRectObj);

    /**
     * Initialize a camshift trachker without calibration.
     * @return 
     */
    CamshiftTracker();

    /**
     * Recalibrate the tracking using a new image and a new object rectangle
     * @param img Image to use for calibration (will not be modified)
     * @param rectObj Rectangle of the object on the given image to use for calibration
     */
    void recalibrate(const cv::Mat &img, const cv::Rect &rectObj);

    /**
     * Track the object on the given image
     * @param t_img Image on which to perform the tracking (will not be modified)
     * @return A rotated rectangle that reprents the best new position found of the tracked object
     */
    cv::RotatedRect trackObj(const cv::Mat &img);

    /**
     * @return the last generated backprof of the tracking method
     */
    const cv::Mat &getBackproj() const;

    int vMin = 10;
    int vMax = 256;
    int sMin = 32;
    int threshold = 160;
    int hsize = 16;

private:
    // Private methods and functions
    void computeColors(const cv::Mat &img);

    // Private fields
    cv::Rect trackWindow;
    cv::Mat hsv, hue, mask, hist, backproj;
};
