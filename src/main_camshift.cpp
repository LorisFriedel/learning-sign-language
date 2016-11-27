//
// @author Loris Friedel
//

#include <tclap/CmdLine.h>
#include <cv.hpp>
#include "../inc/code.h"
#include "../inc/log.h"
#include "../inc/VideoStreamReader.hpp"
#include "../inc/ObjectDetector.hpp"
#include "../inc/CamshiftRunner.hpp"
#include "../inc/KeyInputHandler.hpp"
#include "../inc/colors.h"
#include "../inc/constant.h"

int runCamshiftTrack(VideoStreamReader &vsr, const cv::CascadeClassifier &cascade);

// Main
int main(int argc, const char **argv) {
    try {
        TCLAP::CmdLine cmd("!!! Help for camshift tracking program. It uses a pre-configured classifier for object detect and camshift calibration. !!!"
                                   "\n=> Press '$' to stop the program. "
                                   "\n=> Press '*' to toggle backprojection display."
                                   "\n=> Press '!' to force the tracker to recalibrate using object detection."
                                   "\nWritten by Loris Friedel",
                           ' ', "1.0");

        TCLAP::ValueArg<std::string> cascadeArg("c", "cascade", "Specify the config file to use for the object detection. Default value is " + Default::CASCADE_PATH,
                                                false, Default::CASCADE_PATH, "PATH_TO_XML", cmd);

        TCLAP::ValueArg<std::string> inputArg("i", "input",
                                              "Specify the video input file name (can be a webcam ID, e.g. 0, or a video file, e.g. 'video_test.avi'). Default value is " + Default::INPUT,
                                              false, Default::INPUT, "VIDEO_INPUT_FILE", cmd);
        //// Parse the argv array
        cmd.parse(argc, argv);

        //// Get the value parsed by each arg
        // Load pre-trained cascade data
        std::string &cascadeName = cascadeArg.getValue();
        cv::CascadeClassifier cascade;
        if (!cascade.load(cascadeName)) {
            LOG_E("ERROR: Could not load classifier cascade");
            return Code::CASCADE_LOAD_ERROR;
        }

        // Get file to read image from for image detection
        std::string &inputFilename = inputArg.getValue();
        int webcamId;
        bool isWebcam = false;
        if (inputFilename.empty() || (isdigit(inputFilename[0]) && inputFilename.size() == 1)) {
            webcamId = inputFilename.empty() ? 0 : inputFilename[0] - '0';
            isWebcam = true;
        }


        // Init video reader
        VideoStreamReader vsr;
        int openCodeResult = isWebcam ? vsr.openStream(webcamId) : vsr.openStream(inputFilename);
        if (openCodeResult == Code::ERROR) {
            LOG_E("ERROR: Video capturing failed");
            return Code::ERROR;
        }

        // If image capture has successfully started, start detecting objects regarding mode
        LOG_I("Video capturing has been started ...");

        return runCamshiftTrack(vsr, cascade);

    } catch (TCLAP::ArgException &e) {  // catch any exceptions
        LOG_E("error: " << e.error() << " for arg " << e.argId());
    }

    LOG_E("Program exited with errors");
    return Code::ERROR;
}

/*
 * -------- Camshift track ---------
 */
int runCamshiftTrack(VideoStreamReader &vsr, const cv::CascadeClassifier &cascade) {
    ObjectDetector objectDetector(cascade);
    CamshiftRunner cRunner(vsr, objectDetector);

    // Init slider to tweak thresholds..
    cv::namedWindow("CamShift", 0);
    cv::createTrackbar("Vmin", "CamShift", &cRunner.cTracker.vMin, 256, 0);
    cv::createTrackbar("Vmax", "CamShift", &cRunner.cTracker.vMax, 256, 0);
    cv::createTrackbar("Smin", "CamShift", &cRunner.cTracker.sMin, 256, 0);
    cv::createTrackbar("Brutal threshold", "CamShift", &cRunner.cTracker.threshold, 256, 0);

    // Control variables
    bool backprojDisplay = false;

    // Bind shortcuts
    KeyInputHandler keyHandler;

    std::function<void(const int &)> actionStop([&cRunner](const int &k) {
        cRunner.stop();
    });
    keyHandler.bind('$', &actionStop);

    std::function<void(const int &)> actionRecalibrate([&cRunner](const int &k) {
        cRunner.recalibrate();
        LOG_I("Ask for recalibration");
    });
    keyHandler.bind('!', &actionRecalibrate);

    std::function<void(const int &)> actionBackprojDisplay([&backprojDisplay](const int &k) {
        backprojDisplay = !backprojDisplay;
        LOG_I("Backproj display " << (backprojDisplay ? "enabled" : "disabled"));
    });
    keyHandler.bind('*', &actionBackprojDisplay);

    const std::function<void(CamshiftTracker &, const cv::Mat &, const cv::RotatedRect &, const bool)>
    track_callback([&keyHandler, &backprojDisplay]
                                (CamshiftTracker &cTracker, const cv::Mat &img,
                                 const cv::RotatedRect &objectTracked, const bool objectFound) {
        // Read input key
        int key = cv::waitKey(1);
        keyHandler.apply(key);

        // Drawings and display
        if (backprojDisplay) {
            cv::cvtColor(cTracker.getBackproj(), img, cv::COLOR_GRAY2BGR);
        }

        if (objectFound) {
            cv::ellipse(img, objectTracked, Color::GREEN, 2, CV_AA);
        }
        cv::imshow("CamShift", img);
    });

    return cRunner.runTracking(&track_callback);
}