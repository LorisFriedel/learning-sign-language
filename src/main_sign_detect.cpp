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
#include "../inc/HandTracker.hpp"
#include "../inc/MLPHand.hpp"
#include "../inc/time.h"
#include "../inc/DataYmlWriter.hpp"

int run_camshift_track_hand(VideoStreamReader &vsr, const cv::CascadeClassifier &cascade,
                            const std::string model_path, const std::string image_out_path,
                            const std::string backproj_out_path);

void save_images(const int key, const cv::Mat &img,
                 const CamshiftTracker &c_tracker, const cv::Rect h_rect,
                 const std::string image_out_path,
                 const std::string backproj_out_path);

cv::Mat convert_to_hand_input(const cv::Mat &input, const cv::Rect roi);

cv::Mat crop_resize_flatten(const cv::Mat &input, const cv::Rect &roi);

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
                                              "Specify the video input file name (can be a webcam ID, e.g. 0, or a video file, e.g. 'video_test.avi'). Default value is " + Default::INPUT,
                                              false, Default::INPUT, "VIDEO_INPUT_FILE", cmd);

        TCLAP::ValueArg<std::string> modelArg("m", "model",
                                              "Specify the path to the model to use for sign detection. Default value is " + Default::MODEL_PATH,
                                              false, Default::MODEL_PATH, "PATH_TO_XML_MODEL_FILE", cmd);

        TCLAP::ValueArg<std::string> imageOutputArg("o", "image-output",
                                                    "Specify the path where to save images (.png) when the save image mode is enabled. Default value is " + Default::LETTERS_IMAGES_PATH,
                                                    false, Default::LETTERS_IMAGES_PATH, "DIRECTORY_PATH", cmd);

        TCLAP::ValueArg<std::string> imageBackprojOutputArg("b", "backproj-output",
                                                            "Specify the path where to save backproj data (.yml) when the save image mode is enabled. Default value is " + Default::LETTERS_DATA_PATH,
                                                            false, Default::LETTERS_DATA_PATH, "DIRECTORY_PATH", cmd);
        //// Parse the argv array
        cmd.parse(argc, argv);

        //// Get the value parsed by each arg
        std::string model_path = modelArg.getValue();
        std::string image_output_path = imageOutputArg.getValue();
        std::string backproj_output_path = imageBackprojOutputArg.getValue();

        // Load pre-trained cascade data
        std::string cascade_name = Default::CASCADE_PATH;
        cv::CascadeClassifier cascade;
        if (!cascade.load(cascade_name)) {
            LOG_E("ERROR: Could not load classifier cascade for face");
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

        return run_camshift_track_hand(vsr, cascade, model_path, image_output_path,
                                       backproj_output_path);

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
run_camshift_track_hand(VideoStreamReader &vsr, const cv::CascadeClassifier &cascade,
                        const std::string model_path, const std::string image_out_path,
                        const std::string backproj_out_path) {
    ObjectDetector face_detector(cascade);
    CamshiftRunner c_runner(vsr, face_detector);

    // Init slider to tweak thresholds..
    cv::namedWindow("CamShift", 0);
    cv::createTrackbar("Vmin", "CamShift", &c_runner.c_tracker.v_min, 256, 0);
    cv::createTrackbar("Vmax", "CamShift", &c_runner.c_tracker.v_max, 256, 0);
    cv::createTrackbar("Smin", "CamShift", &c_runner.c_tracker.s_min, 256, 0);
    cv::createTrackbar("Brutal threshold", "CamShift", &c_runner.c_tracker.threshold, 256, 0);

    // Create hand tracker
    HandTracker h_tracker;

    // Load model
    MLPHand mlp_hand;
    mlp_hand.learn_from(model_path);

    // Variables for hand tracking
    cv::RotatedRect hand_tracked;
    bool hand_found;

    // Control variables
    bool backproj_display = false;
    bool save_img_enable = false;

    const std::function<void(CamshiftTracker &, const cv::Mat &, const cv::RotatedRect &, const bool)>
            track_face_callback([&](CamshiftTracker &c_tracker, const cv::Mat &img,
                                    const cv::RotatedRect &face_tracked, const bool face_found) {
        if (face_found) {
            // Track hand
            hand_tracked = h_tracker.track_hand(img, c_tracker.get_backproj(), face_tracked.boundingRect());
            hand_found = hand_tracked.boundingRect().area() > 1;

            // Handle result
            if (hand_found) {
                // TODO hand found
            } else {
                //recalibrate = true;
            }
        } else {
            hand_found = false;
        }

        // Read input key
        int key = cv::waitKey(1);
        switch (key) {
            case '$':
                c_runner.stop();
                break;
            case '!':
                c_runner.recalibrate();
                LOG_I("Ask for recalibration");
                break;
            case '*':
                backproj_display = !backproj_display;
                LOG_I("Backproj display " << (backproj_display ? "enabled" : "disabled"));
                break;
            case ':':
                save_img_enable = !save_img_enable;
                LOG_I("Save image mode " << (save_img_enable ? "enabled" : "disabled"));
                break;
            default:
                if (save_img_enable && ('a' <= key && key <= 'z')) {
                    save_images(key, img, c_tracker, hand_tracked.boundingRect(),
                                image_out_path, backproj_out_path);
                }
                break;
        }

        // Drawings and display
        if (backproj_display) {
            cv::cvtColor(c_tracker.get_backproj(), img, cv::COLOR_GRAY2BGR);
        }

        if (hand_found) {
            // Prediction
            cv::Mat small_backproj = convert_to_hand_input(c_tracker.get_backproj(), hand_tracked.boundingRect());


//            // TODO ----
//            cv::Rect roi = hand_tracked.boundingRect();
//            int largest = roi.height > roi.width ? roi.height : roi.width;
//            cv::Rect roi_tmp = cv::Rect(roi.x, roi.y, largest, largest);
//            cv::Rect img_rect(0, 0, img.cols, img.rows);
//            cv::Rect roi_final = roi_tmp & img_rect;
//            if (roi_final.width > roi_final.height) {
//                roi_final.width = roi_final.height;
//            } else if (roi_final.width < roi_final.height) {
//                roi_final.height = roi_final.width;
//            }
//            std::vector<float> description;
//            cv::HOGDescriptor descriptor(cv::Size(16, 16), cv::Size(4, 4), cv::Size(2, 2), cv::Size(2, 2), 9);
//            cv::Mat cropped = c_tracker.get_backproj()(roi_final);
//            descriptor.compute(cropped, description);
//            std::cout << description.size() << " : ";
//            // TODO ----

            std::pair<int, float> mlp_prediction = mlp_hand.predict(small_backproj);

            if (mlp_prediction.second > 0.5) {
                std::stringstream text_prediction;
                text_prediction << "Letter: " << ((char) (mlp_prediction.first + 'a'))
                                << " - Proba: " << mlp_prediction.second * 100 << "%";
                cv::putText(img, text_prediction.str(), cvPoint(32, 32), cv::QT_FONT_NORMAL, 0.8, Color::WHITE);
            }

            // Drawing face and hand
            cv::ellipse(img, face_tracked, Color::RED, 2, CV_AA);
            cv::ellipse(img, hand_tracked, Color::BLUE, 2, CV_AA);
            cv::Rect r(hand_tracked.boundingRect());
            rectangle(img, cvPoint(cvRound(r.x), cvRound(r.y)),
                      cvPoint(cvRound((r.x + r.width - 1)), cvRound((r.y + r.height - 1))),
                      Color::GREEN, 2, 8, 0);
        }
        cv::imshow("CamShift", img);
    });

    return c_runner.run_tracking(&track_face_callback);
}


cv::Mat convert_to_hand_input(const cv::Mat &input, const cv::Rect roi) {
    int largest = roi.height > roi.width ? roi.height : roi.width;
    cv::Rect roi_tmp = cv::Rect(roi.x, roi.y, largest, largest);
    cv::Rect img_rect(0, 0, input.cols, input.rows);
    cv::Rect roi_final = roi_tmp & img_rect;

    if (roi_final.width > roi_final.height) {
        roi_final.width = roi_final.height;
    } else if (roi_final.width < roi_final.height) {
        roi_final.height = roi_final.width;
    }

    cv::Mat result = crop_resize_flatten(input, roi_final); // crop + resize + flatten
    result.convertTo(result, CV_32FC1, 1.0 / 255.0);

    return result;
}

cv::Mat crop_resize_flatten(const cv::Mat &input, const cv::Rect &roi) {
    cv::Mat result;
    cv::resize(input(roi), result, cv::Size(16, 16)); // crop + resize
    result = result.reshape(0, 1); // flatten
    return result;
}

void    save_images(const int key, const cv::Mat &img,
                 const CamshiftTracker &c_tracker, const cv::Rect h_rect,
                 const std::string image_out_path,
                 const std::string backproj_out_path) {
    LOG_I("Saving image for key \'" << (char) key << "\' ...");

    // Get the largest value between height and width (maximizing probabilities to keep the whole hand)
    // Then create the final safe roi to use
    int largest = h_rect.height > h_rect.width ? h_rect.height : h_rect.width;
    cv::Rect roi = cv::Rect(h_rect.x, h_rect.y, largest, largest);
    cv::Rect img_rect(0, 0, img.cols, img.rows);
    cv::Rect roi_final = roi & img_rect; // safe part

    // Crop image
    cv::Mat img_cropped = img(roi_final);

    // Create save path
    std::stringstream file_name;
    file_name << std::string(1, key) << "_fl300459_" << std::to_string(get_timestamp());

    // Save cropped image
    std::stringstream image_file_path;
    image_file_path << image_out_path << file_name.str() << ".png";
    cv::imwrite(image_file_path.str(), img_cropped);

    // Crop, resize and flatten backproj
    cv::Mat small_backproj = crop_resize_flatten(c_tracker.get_backproj(), roi_final); // crop

    // Save small backproj to yml file
    std::stringstream data_file_path;
    data_file_path << backproj_out_path << file_name.str() << ".yml";

    DataYmlWriter writer(data_file_path.str());
    writer.write(small_backproj, key);
}
