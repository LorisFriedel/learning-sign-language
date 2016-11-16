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
#include "../inc/DataYmlWriter.hpp"
#include "../inc/colors.h"
#include "../inc/constant.h"
#include "../inc/HandTracker.hpp"
#include "../inc/MLPHand.hpp"
#include "../inc/time.h"

int runCamshiftTrackHand(VideoStreamReader &vsr, const cv::CascadeClassifier &cascade,
                         const std::string modelPath, const std::string imageOutPath,
                         const std::string backprojOutPath);

void saveImages(const int key, const cv::Mat &img,
                const CamshiftTracker &cTracker, const cv::Rect hRect,
                const std::string imageOutPath,
                const std::string backprojOutPath);

cv::Mat convertToHandInput(const cv::Mat &input, const cv::Rect roi);

cv::Mat cropResizeFlatten(const cv::Mat &input, const cv::Rect &roi);

int main(int argc, const char **argv) {
    try {
        TCLAP::CmdLine cmd(
                "Help for camshift tracking program. It uses a pre-configured classifier for object detect and camshift calibration."
                        "\n"
                        "\nBe careful, every default directory value"
                        "\n"
                        "\n=> Press '$' to stop the program. "
                        "\n=> Press '*' to toggle backprojection display."
                        "\n=> Press '!' to force the tracker to recalibrate using object detection."
                        "\n=> Press ':' to toggle save image mode."
                        "\n=> Press any letter to save image and backproj data when save image mode is enabled."
                        "\nWritten by Loris Friedel",
                ' ', "1.0");

        TCLAP::ValueArg<std::string> inputArg("i", "input",
                                              "Specify the video input file name (can be a webcam ID, e.g. 0, or a video file, e.g. 'video_test.avi'). Default value is " +
                                              Default::INPUT,
                                              false, Default::INPUT, "VIDEO_INPUT_FILE", cmd);

        TCLAP::ValueArg<std::string> modelArg("m", "model",
                                              "Specify the path to the model to use for sign detection. Default value is " +
                                              Default::MODEL_PATH,
                                              false, Default::MODEL_PATH, "PATH_TO_XML_MODEL_FILE", cmd);

        TCLAP::ValueArg<std::string> imageOutputArg("o", "image-output",
                                                    "Specify the path where to save images (.png) when the save image mode is enabled. Default value is " +
                                                    Default::LETTERS_IMAGES_PATH,
                                                    false, Default::LETTERS_IMAGES_PATH, "DIRECTORY_PATH", cmd);

        TCLAP::ValueArg<std::string> imageBackprojOutputArg("b", "backproj-output",
                                                            "Specify the path where to save backproj data (.yml) when the save image mode is enabled. Default value is " +
                                                            Default::LETTERS_DATA_PATH,
                                                            false, Default::LETTERS_DATA_PATH, "DIRECTORY_PATH", cmd);

        // TODO : mode HOG for prediction

        //// Parse the argv array
        cmd.parse(argc, argv);

        //// Get the value parsed by each arg
        std::string modelPath = modelArg.getValue();
        std::string imageOutputPath = imageOutputArg.getValue();
        std::string backprojOutputPath = imageBackprojOutputArg.getValue();

        // Load pre-trained cascade data
        std::string cascadeName = Default::CASCADE_PATH;
        cv::CascadeClassifier cascade;
        if (!cascade.load(cascadeName)) {
            LOG_E("ERROR: Could not load classifier cascade for face");
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

        // Init video reader
        VideoStreamReader vsr;
        int openCodeResult = isWebcam ? vsr.openStream(webcamId) : vsr.openStream(inputFilename);
        if (openCodeResult == Code::ERROR) {
            LOG_E("ERROR: Video capturing failed");
            return Code::ERROR;
        }

        // If image capture has successfully started, start detecting objects regarding mode
        LOG_I("Video capturing has been started ...");

        return runCamshiftTrackHand(vsr, cascade, modelPath, imageOutputPath, backprojOutputPath);

    } catch (TCLAP::ArgException &e) {  // catch any exceptions
        LOG_E("error: " << e.error() << " for arg " << e.argId());
    }

    LOG_E("Program exited with errors");
    return Code::ERROR;
}


/*
 * -------- Camshift HAND ---------
 */
int
runCamshiftTrackHand(VideoStreamReader &vsr, const cv::CascadeClassifier &cascade,
                     const std::string modelPath, const std::string imageOutPath,
                     const std::string backprojOutPath) {
    ObjectDetector faceDetector(cascade);
    CamshiftRunner cRunner(vsr, faceDetector);

    // Init slider to tweak thresholds..
    cv::namedWindow("CamShift", 0);
    cv::createTrackbar("Vmin", "CamShift", &cRunner.cTracker.vMin, 256, 0);
    cv::createTrackbar("Vmax", "CamShift", &cRunner.cTracker.vMax, 256, 0);
    cv::createTrackbar("Smin", "CamShift", &cRunner.cTracker.sMin, 256, 0);
    cv::createTrackbar("Brutal threshold", "CamShift", &cRunner.cTracker.threshold, 256, 0);

    // Create hand tracker
    HandTracker hTracker;

    // Load model
    MLPHand mlpHand;
    mlpHand.learnFrom(modelPath);

    // Variables for hand tracking
    cv::RotatedRect handTracked;
    bool handFound;

    // Control variables
    bool backprojDisplay = false;
    bool saveImgEnable = false;

    const std::function<void(CamshiftTracker &, const cv::Mat &, const cv::RotatedRect &, const bool)>
            trackFaceCallback([&](CamshiftTracker &cTracker, const cv::Mat &img,
                                    const cv::RotatedRect &faceTracked, const bool faceFound) {
        if (faceFound) {
            // Track hand
            handTracked = hTracker.trackHand(img, cTracker.getBackproj(), faceTracked.boundingRect());
            handFound = handTracked.boundingRect().area() > 1;

            // Handle result
            if (handFound) {
                // TODO hand found
            } else {
                //recalibrate = true;
            }
        } else {
            handFound = false;
        }

        // Read input key
        int key = cv::waitKey(1);
        switch (key) {
            case '$':
                cRunner.stop();
                break;
            case '!':
                cRunner.recalibrate();
                LOG_I("Ask for recalibration");
                break;
            case '*':
                backprojDisplay = !backprojDisplay;
                LOG_I("Backproj display " << (backprojDisplay ? "enabled" : "disabled"));
                break;
            case ':':
                saveImgEnable = !saveImgEnable;
                LOG_I("Save image mode " << (saveImgEnable ? "enabled" : "disabled"));
                break;
            default:
                if (saveImgEnable && ('a' <= key && key <= 'z')) {
                    saveImages(key, img, cTracker, handTracked.boundingRect(),
                               imageOutPath, backprojOutPath);
                }
                break;
        }

        // Drawings and display
        if (backprojDisplay) {
            cv::cvtColor(cTracker.getBackproj(), img, cv::COLOR_GRAY2BGR);
        }

        if (handFound) {
            // Prediction
            cv::Mat smallBackproj = convertToHandInput(cTracker.getBackproj(), handTracked.boundingRect());

            std::pair<int, float> mlpPrediction = mlpHand.predict(smallBackproj);

            if (mlpPrediction.second > 0.5) {
                std::stringstream textPrediction;
                textPrediction << "Letter: " << ((char) (mlpPrediction.first + 'a'))
                                << " - Proba: " << mlpPrediction.second * 100 << "%";
                cv::putText(img, textPrediction.str(), cvPoint(32, 32), cv::QT_FONT_NORMAL, 0.8, Color::WHITE);
            }

            // Drawing face and hand
            cv::ellipse(img, faceTracked, Color::RED, 2, CV_AA);
            cv::ellipse(img, handTracked, Color::BLUE, 2, CV_AA);
            cv::Rect r(handTracked.boundingRect());
            rectangle(img, cvPoint(cvRound(r.x), cvRound(r.y)),
                      cvPoint(cvRound((r.x + r.width - 1)), cvRound((r.y + r.height - 1))),
                      Color::GREEN, 2, 8, 0);
        }
        cv::imshow("CamShift", img);
    });

    return cRunner.runTracking(&trackFaceCallback);
}


cv::Mat convertToHandInput(const cv::Mat &input, const cv::Rect roi) {
    int largest = roi.height > roi.width ? roi.height : roi.width;
    cv::Rect roiTmp = cv::Rect(roi.x, roi.y, largest, largest);
    cv::Rect imgRect(0, 0, input.cols, input.rows);
    cv::Rect roiFinal = roiTmp & imgRect;

    if (roiFinal.width > roiFinal.height) {
        roiFinal.width = roiFinal.height;
    } else if (roiFinal.width < roiFinal.height) {
        roiFinal.height = roiFinal.width;
    }

    cv::Mat result = cropResizeFlatten(input, roiFinal); // crop + resize + flatten
    result.convertTo(result, CV_32FC1, 1.0 / 255.0);

    return result;
}

cv::Mat cropResizeFlatten(const cv::Mat &input, const cv::Rect &roi) {
    cv::Mat result;
    cv::resize(input(roi), result, cv::Size(16, 16)); // crop + resize
    result = result.reshape(0, 1); // flatten
    return result;
}

void saveImages(const int key, const cv::Mat &img,
                const CamshiftTracker &cTracker, const cv::Rect hRect,
                const std::string imageOutPath,
                const std::string backprojOutPath) {
    LOG_I("Saving image for key \'" << (char) key << "\' ...");

    // Get the largest value between height and width (maximizing probabilities to keep the whole hand)
    // Then create the final safe roi to use
    int largest = hRect.height > hRect.width ? hRect.height : hRect.width;
    cv::Rect roi = cv::Rect(hRect.x, hRect.y, largest, largest);
    cv::Rect imgRect(0, 0, img.cols, img.rows);
    cv::Rect roiFinal = roi & imgRect; // safe part

    // Crop image
    cv::Mat imgCropped = img(roiFinal);

    // Create save path
    std::stringstream fileName;
    fileName << std::string(1, key) << "_fl300459_" << std::to_string(get_timestamp());

    // Save cropped image
    std::stringstream imageFilePath;
    imageFilePath << imageOutPath << fileName.str() << ".png";
    cv::imwrite(imageFilePath.str(), imgCropped);

    // Crop, resize and flatten backproj
    cv::Mat smallBackproj = cropResizeFlatten(cTracker.getBackproj(), roiFinal); // crop

    // Save small backproj to yml file
    std::stringstream dataFilePath;
    dataFilePath << backprojOutPath << fileName.str() << ".yml";

    DataYmlWriter writer(dataFilePath.str());
    writer.write(smallBackproj, key);
}
