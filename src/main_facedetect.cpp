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
        std::string cascadeName = cascadeArg.getValue();
        cv::CascadeClassifier cascade;
        if (!cascade.load(cascadeName)) {
            LOG_E("ERROR: Could not load classifier cascade");
            return Code::CASCADE_LOAD_ERROR;
        }

        // Get file to read image from for image detection
        std::string inputFilename = inputArg.getValue();
        int webcamId;
        bool isWebcam = false;
        if (inputFilename.empty() || (isdigit(inputFilename[0]) && inputFilename.size() == 1)) {
            webcamId = inputFilename.empty() ? 0 : inputFilename[0] - '0';
            isWebcam = true;
        }

        // Handling mode
        // Init video reader
        VideoStreamReader vsr;
        int openCodeResult = isWebcam ? vsr.openStream(webcamId) : vsr.openStream(inputFilename);
        if (openCodeResult == Code::ERROR) {
            LOG_E("ERROR: Video capturing failed");
            return Code::ERROR;
        }

        // If image capture has successfully started, start detecting objects regarding mode
        LOG_I("Video capturing has been started ...");

        // q pour quitter
        ObjectDetectRunner detectorRunner(vsr, cascade);
        return detectorRunner.runDetection();
    } catch (TCLAP::ArgException &e) {  // catch any exceptions
        LOG_E("error: " << e.error() << " for arg " << e.argId());
    }

    LOG_E("Program exited with errors");
    return Code::ERROR;
}
