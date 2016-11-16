//
// Created by loris on 9/19/16.
//

#include "../inc/ObjectDetector.hpp"
#include "../inc/VideoStreamReader.hpp"
#include "../inc/log.h"
#include "../inc/geo.h"

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video/tracking.hpp"

namespace {
    const int DETECT_SQUARE_SIZE = 160;
    const int MARGIN_DETECT = 32;
    const int MARGIN_IGNORE_MOVE = 16;
}

ObjectDetector::ObjectDetector(const cv::CascadeClassifier &cascade)
        : cascade(cascade) {
}

const std::tuple<bool, double, cv::Rect>
ObjectDetector::detectIn(const cv::Mat &img) {
    cv::Mat gray, smallImg;

    // Convert to grayscale and resize image for performance improvements
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    resize(gray, smallImg, cv::Size(), 1, 1, cv::INTER_LINEAR);
    equalizeHist(smallImg, smallImg);

    // Vector to store result (will contains most likely a single value)
    std::vector<cv::Rect> objects;

    // Object detection (by opencv)
    double t = (double) cvGetTickCount();
    cascade.detectMultiScale(smallImg, objects,
                              1.1, 2, 0
                                      //|CASCADE_FIND_BIGGEST_OBJECT|CASCADE_DO_ROUGH_SEARCH
                                      | cv::CASCADE_SCALE_IMAGE,
                              cv::Size(DETECT_SQUARE_SIZE, DETECT_SQUARE_SIZE));

    // Detection time
    t = (double) cvGetTickCount() - t;
    const double detectTimeMs = (t / (cvGetTickFrequency() * 1000.));

    // Result handling
    bool detected = false;
    cv::Rect result = Geo::NULL_RECT;
    if (objects.size() >= 1) {
        detected = true;
        result = objects[0];
    }

    return std::tuple<bool, double, cv::Rect>(detected, detectTimeMs, result);
};

const std::tuple<bool, double, cv::Rect>
ObjectDetector::detect(const cv::Mat &img, const cv::Rect &roi) {
    cv::Mat imgClone(img.clone(), roi);

    std::tuple<bool, double, cv::Rect> result = detectIn(imgClone);
    cv::Rect &detectedRect = std::get<2>(result);

    if (std::get<1>(result)) {
        // Fix result origin
        detectedRect.x += roi.x;
        detectedRect.y += roi.y;

        if (detectedRect.x < 0) { detectedRect.x = 0; }
        if (detectedRect.y < 0) { detectedRect.y = 0; }
    }

    return result;
}

const bool shouldIgnoreMove(const cv::Rect &oldRect, const cv::Rect &newRect) {
    return abs(oldRect.x - newRect.x) < MARGIN_IGNORE_MOVE
           && abs(oldRect.y - newRect.y) < MARGIN_IGNORE_MOVE
           && abs(oldRect.width - newRect.width) < MARGIN_IGNORE_MOVE
           && abs(oldRect.height - newRect.height) < MARGIN_IGNORE_MOVE;
}

void ObjectDetector::loopDetect(VideoStreamReader &vsr,
                                const std::function<void(cv::Mat, bool, double, cv::Rect)> &callback,
                                const bool stabilized) {
    cv::Mat img;
    cv::Rect roi = Geo::NULL_RECT;
    cv::Rect roiAugmented;

    img = vsr.readFrame(); // Read the first image (sometimes it's a flipped image..) to init default values
    const cv::Rect roiDefault(0, 0, img.cols, img.rows);

    loopActive = true;
    while (loopActive) {
        img = vsr.readFrame();

        if (roi.area() != 0) {
            // Augment the given region of interest to improve performance
            // (start detection in a smaller area defined by the previous match and only a bit larger)
            const int augX = roi.x - MARGIN_DETECT, augY = roi.y - MARGIN_DETECT;
            const int augWidth = roi.width + (2 * MARGIN_DETECT), augHeight = roi.height + (2 * MARGIN_DETECT);
            roiAugmented = cv::Rect(augX, augY, augWidth, augHeight) & cv::Rect(0, 0, img.cols, img.rows);
        } else {
            roiAugmented = roiDefault;
        }

        std::tuple<bool, double, cv::Rect> result = detect(img, roiAugmented);
        if (stabilized && shouldIgnoreMove(roi, std::get<2>(result))) {
            // Small move
            // Do nothing
        } else {
            // Big move
            roi = std::get<2>(result);
        }
        callback(img, std::get<0>(result), std::get<1>(result), roi);
    }
}

void ObjectDetector::stopLoopDetect() {
    loopActive = false;
}
