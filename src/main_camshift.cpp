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

int run_camshift_track(VideoStreamReader &vsr, const cv::CascadeClassifier &cascade);

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
        std::string cascade_name = cascadeArg.getValue();
        cv::CascadeClassifier cascade;
        if (!cascade.load(cascade_name)) {
            LOG_E("ERROR: Could not load classifier cascade");
            return Code::CASCADE_LOAD_ERROR;
        }

        // Get file to read image from for image detection
        std::string input_filename = inputArg.getValue();
        int webcam_id;
        bool is_webcam = false;
        if (input_filename.empty() || (isdigit(input_filename[0]) && input_filename.size() == 1)) {
            webcam_id = input_filename.empty() ? 0 : input_filename[0] - '0';
            is_webcam = true;
        }


        // Init video reader
        VideoStreamReader vsr;
        int open_code_result = is_webcam ? vsr.openStream(webcam_id) : vsr.openStream(input_filename);
        if (open_code_result == Code::ERROR) {
            LOG_E("ERROR: Video capturing failed");
            return Code::ERROR;
        }

        // If image capture has successfully started, start detecting objects regarding mode
        LOG_I("Video capturing has been started ...");

        return run_camshift_track(vsr, cascade);

    } catch (TCLAP::ArgException &e) {  // catch any exceptions
        LOG_E("error: " << e.error() << " for arg " << e.argId());
    }

    LOG_E("Program exited with errors");
    return Code::ERROR;
}

/*
 * -------- Camshift track ---------
 */
int run_camshift_track(VideoStreamReader &vsr, const cv::CascadeClassifier &cascade) {
    ObjectDetector object_detector(cascade);
    CamshiftRunner c_runner(vsr, object_detector);

    // Init slider to tweak thresholds..
    cv::namedWindow("CamShift", 0);
    cv::createTrackbar("Vmin", "CamShift", &c_runner.c_tracker.v_min, 256, 0);
    cv::createTrackbar("Vmax", "CamShift", &c_runner.c_tracker.v_max, 256, 0);
    cv::createTrackbar("Smin", "CamShift", &c_runner.c_tracker.s_min, 256, 0);
    cv::createTrackbar("Brutal threshold", "CamShift", &c_runner.c_tracker.threshold, 256, 0);

    // Control variables
    bool backproj_display = false;

    // Bind shortcuts
    KeyInputHandler key_handler;

    std::function<void(const int &)> action_stop([&c_runner](const int &k) {
        c_runner.stop();
    });
    key_handler.bind('$', &action_stop);

    std::function<void(const int &)> action_recalibrate([&c_runner](const int &k) {
        c_runner.recalibrate();
        LOG_I("Ask for recalibration");
    });
    key_handler.bind('!', &action_recalibrate);

    std::function<void(const int &)> action_backproj_display([&backproj_display](const int &k) {
        backproj_display = !backproj_display;
        LOG_I("Backproj display " << (backproj_display ? "enabled" : "disabled"));
    });
    key_handler.bind('*', &action_backproj_display);

    const std::function<void(CamshiftTracker &, const cv::Mat &, const cv::RotatedRect &, const bool)>
    track_callback([&key_handler, &backproj_display]
                                (CamshiftTracker &c_tracker, const cv::Mat &img,
                                 const cv::RotatedRect &object_tracked, const bool object_found) {
        // Read input key
        int key = cv::waitKey(1);
        key_handler.apply(key);

        // Drawings and display
        if (backproj_display) {
            cv::cvtColor(c_tracker.get_backproj(), img, cv::COLOR_GRAY2BGR);
        }

        if (object_found) {
            cv::ellipse(img, object_tracked, Color::GREEN, 2, CV_AA);
        }
        cv::imshow("CamShift", img);
    });

    return c_runner.run_tracking(&track_callback);
}