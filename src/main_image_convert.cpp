//
// @author Loris Friedel
//

#include <tclap/CmdLine.h>
#include <cv.hpp>
#include <dirent.h>
#include "../inc/code.h"
#include "../inc/log.h"
#include "../inc/constant.h"
#include "../inc/time.h"
#include "../inc/DataYmlReader.hpp"
#include "../inc/DirectoryReader.hpp"
#include "../inc/DataYmlWriter.hpp"


int main(int argc, const char **argv) {
    try {
        TCLAP::CmdLine cmd(
                "!!! Help for image conversion !!!"
                        "\nThis program is made to convert images to HOG data (from grayscale)"
                        "\nWritten by Loris Friedel",
                ' ', "1.0");

        TCLAP::ValueArg<std::string> inputDirArg("i", "input-dir",
                                                 "Directory where .png images are located."
                                                         " Will be converted to .yml of the HOG of the grayscale of the image. ",
                                                 true, ".", "DIRECTORY_PATH", cmd);

        TCLAP::ValueArg<std::string> outputDirArg("o", "output-dir",
                                                  "Directory where to save output data.",
                                                  true, ".", "DIRECTORY_PATH", cmd);

        //// Parse the argv array
        cmd.parse(argc, argv);

        //// Get the value parsed by each arg and handle them
        std::string input = inputDirArg.getValue();
        std::string output = outputDirArg.getValue();

        DirectoryReader dirReader(input);
        dirReader.foreachFile([&output](std::string filePath, std::string fileName){
            // Read image (.png) & convert to grayscale
            cv::Mat image = cv::imread(filePath, CV_LOAD_IMAGE_GRAYSCALE);

            // Resize
            cv::resize(image, image, cv::Size(256,256));

            // Convert to HOG
            std::vector<float> description;
            cv::HOGDescriptor descriptor(cv::Size(16, 16), cv::Size(4, 4), cv::Size(2, 2), cv::Size(2, 2), 9);
            descriptor.compute(image, description);

            // Write HOG
            std::stringstream outPath;
            outPath << output << "/" << fileName;
            DataYmlWriter writer(outPath.str());
            int letter =static_cast<int>(fileName[0]);
            writer.write(description, letter);
        });

    } catch (TCLAP::ArgException &e) {  // catch any exceptions
        LOG_E("error: " << e.error() << " for arg " << e.argId());
    }

    LOG_E("Program exited with errors");
    return Code::ERROR;
}