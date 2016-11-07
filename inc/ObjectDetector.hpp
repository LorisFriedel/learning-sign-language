//
// Created by loris on 9/19/16.
//

#pragma once

#include <functional>
#include <cv.hpp>

class VideoStreamReader;

class ObjectDetector {

public:
    ObjectDetector(const cv::CascadeClassifier &cascade);

    /**
     * Detect a face in the given image.
     *
     * @param img Image where to detect face (will not be modified)
     * @return A tuple <0> true if face detected, false otherwise; <1> detection duration in ms; <2> rectangle of the face area if detected, empty rectangle otherwise
     */
    const std::tuple<bool, double, cv::Rect>
    detect_in(const cv::Mat &img);

    /**
     * Detect a face in the given image restricted to the given region of interest
     *
     * @param img Image where to detect face (will not be modified)
     * @param roi Region of interest.
     * @return A tuple: <0> true if face detected, false otherwise; <1> detection duration in ms; <2> rectangle of the face area if detected, empty rectangle otherwise
     */
    const std::tuple<bool, double, cv::Rect>
    detect(const cv::Mat &img, const cv::Rect &roi);

    /**
     * Using the given video stream reader, this method read a frame
     * then perform a face detection before calling the given method with the result as parameters.
     * The read image will not be modified.
     *
     * @param vsr Image provider
     * @param callback Method to be called with face detection results after each detection performed
     * @param stabilized True if the detection result should be stabilized, false otherwise
     */
    void loop_detect(VideoStreamReader &vsr,
                     const std::function<void(cv::Mat, bool, double, cv::Rect)> &callback,
                     const bool stabilized = true);

    /**
     * Stop a previously started loop detection
     */
    void stop_loop_detect();

private:
    cv::CascadeClassifier _cascade;
    bool _loop_active = false;
};
