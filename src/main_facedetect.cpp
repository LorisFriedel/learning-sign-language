//
// @author Loris Friedel
//

#include <cv.hpp>
#include <tclap/CmdLine.h>
#include "../inc/code.h"
#include "../inc/log.h"
#include "../inc/VideoStreamReader.hpp"
#include "../inc/ObjectDetectRunner.hpp"
#include "../inc/constant.h"

// Main
int main(int argc, const char **argv) {
    try {
        TCLAP::CmdLine cmd("!!! Help for object detection program using pre-configured classifier. !!!"
                                   "\n=> Press 'q' to stop running object detection. "
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

        // Handling mode
        // Init video reader
        VideoStreamReader vsr;
        int open_code_result = is_webcam ? vsr.openStream(webcam_id) : vsr.openStream(input_filename);
        if (open_code_result == Code::ERROR) {
            LOG_E("ERROR: Video capturing failed");
            return Code::ERROR;
        }

        // If image capture has successfully started, start detecting objects regarding mode
        LOG_I("Video capturing has been started ...");

        // q pour quitter
        ObjectDetectRunner detector_runner(vsr, cascade);
        return detector_runner.run_detection();
    } catch (TCLAP::ArgException &e) {  // catch any exceptions
        LOG_E("error: " << e.error() << " for arg " << e.argId());
    }

    LOG_E("Program exited with errors");
    return Code::ERROR;
}
