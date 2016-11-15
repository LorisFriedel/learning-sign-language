//
// @author Loris Friedel
//

#include <tclap/CmdLine.h>
#include <cv.hpp>
#include <dirent.h>
#include <stdlib.h>
#include "../inc/code.h"
#include "../inc/log.h"
#include "../inc/constant.h"
#include "../inc/time.h"
#include "../inc/DataYmlReader.hpp"
#include "../inc/DirectoryReader.hpp"
#include "../inc/DataYmlWriter.hpp"
#include "../inc/Timer.hpp"


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

        TCLAP::ValueArg<int> imgSizeArg("s", "image-size",
                                        "Specify the target size of the image. Images are resized to be a square. Default value is " +
                                        std::to_string(Default::HOG_IMG_SIZE),
                                        false, Default::HOG_IMG_SIZE, "POSITIVE_INTEGER", cmd);

        TCLAP::ValueArg<int> blockSizeArg("b", "block-size",
                                          "Specify the block size for HOG description. Default value is " +
                                          std::to_string(Default::HOG_BLOCK_SIZE),
                                          false, Default::HOG_BLOCK_SIZE, "POSITIVE_INTEGER", cmd);

        TCLAP::ValueArg<int> blockStrideSizeArg("k", "block-stride",
                                                "Specify the block stride size for HOG description. Default value is " +
                                                std::to_string(Default::HOG_BLOCK_STRIDE_SIZE),
                                                false, Default::HOG_BLOCK_STRIDE_SIZE, "POSITIVE_INTEGER", cmd);

        TCLAP::ValueArg<int> cellSizeArg("c", "cell-size",
                                         "Specify the cell size for HOG description. Default value is " +
                                         std::to_string(Default::HOG_CELL_SIZE),
                                         false, Default::HOG_CELL_SIZE, "POSITIVE_INTEGER", cmd);

        TCLAP::SwitchArg verboseArg("v", "verbose",
                                    "Log everything.",
                                    cmd, false);

        //// Parse the argv array
        cmd.parse(argc, argv);

        //// Get the value parsed by each arg and handle them
        std::string input = inputDirArg.getValue();
        std::string output = outputDirArg.getValue();
        int imgSize = imgSizeArg.getValue();
        int blockSize = blockSizeArg.getValue();
        int blockStrideSize = blockStrideSizeArg.getValue();
        int cellSize = cellSizeArg.getValue();
        bool verbose = verboseArg.getValue();

        Timer globalMonitor, readGrayTimer,
                resizeTimer, hogTimer, writeTimer;
        double nbOfImage = 0;
        double readGrayTotal = 0;
        double resizeTotal = 0;
        double hogTotal = 0;
        double writeTotal = 0;

        // Start timer
        globalMonitor.start();

        std::stringstream cmdMkdir;
        cmdMkdir << "mkdir -p " << output;
        system(cmdMkdir.str().c_str());

        LOG_I("Conversion in progress...");
        DirectoryReader dirReader(input);
        dirReader.foreachFile([&](std::string filePath, std::string fileName) {
            // Read image (.png) & convert to grayscale
            if (verbose) {
                nbOfImage++;
                LOG_I("(" << fileName << ") Reading image & converting it to grayscale..");
                readGrayTimer.start();
            }
            cv::Mat image = cv::imread(filePath, CV_LOAD_IMAGE_GRAYSCALE);
            if (verbose) {
                readGrayTimer.stop();
                readGrayTotal += readGrayTimer.getDurationMS();
            }

            // Resize
            if (verbose) {
                LOG_I("Resizing image (" << imgSize << "x" << imgSize << ")..");
                resizeTimer.start();
            }
            cv::resize(image, image, cv::Size(imgSize, imgSize));
            if (verbose) {
                resizeTimer.stop();
                resizeTotal += resizeTimer.getDurationMS();
            }

            // Convert to HOG
            if (verbose) {
                LOG_I("Compute HOG " << "(block:" << blockSize
                                     << ", block stride:" << blockStrideSize
                                     << ", cell:" << cellSize
                                     << ") ..");
                hogTimer.start();
            }

            std::vector<float> description;
            cv::HOGDescriptor descriptor(cv::Size(imgSize, imgSize),
                                         cv::Size(blockSize, blockSize),
                                         cv::Size(blockStrideSize, blockStrideSize),
                                         cv::Size(cellSize, cellSize),
                                         9);
            descriptor.compute(image, description);
            if (verbose) {
                hogTimer.stop();
                hogTotal += hogTimer.getDurationMS();
            }

            // Create new path for data
            std::stringstream outPath;
            fileName = fileName.substr(0, fileName.length() - 3); // remove extension (.png)
            outPath << output << "/" << fileName << "yml"; // create final path for .yml files

            // Write HOG
            if (verbose) {
                LOG_I("(" << outPath.str() << ") Write image..");
                writeTimer.start();
            }

            int letter, i;
            for (i = 0; !(97 <= fileName[i] && fileName[i] <= 122); i++);
            letter = static_cast<int>(fileName[i]);

            DataYmlWriter writer(outPath.str());
            writer.write(description, letter);
            if (verbose) {
                writeTimer.stop();
                writeTotal += writeTimer.getDurationMS();
            }
        });

        // End timer
        globalMonitor.stop();

        LOG_I("Conversion done! (" << globalMonitor.getDurationS() << " s)");

        if (verbose) {
            LOG_I("Average for each image processing: "
                          << (readGrayTotal + resizeTotal + hogTotal + writeTotal) / (nbOfImage) << " ms");
            LOG_I("Average time for each action:");
            LOG_I(" - Read and convert image to grayscale: " << (readGrayTotal / nbOfImage) << " ms");
            LOG_I(" - Resize image: " << (resizeTotal / nbOfImage) << " ms");
            LOG_I(" - Compute HOG: " << (hogTotal / nbOfImage) << " ms");
            LOG_I(" - Write data on disk: " << (writeTotal / nbOfImage) << " ms");
        }

        return Code::SUCCESS;
    } catch (TCLAP::ArgException &e) {  // catch any exceptions
        LOG_E("error: " << e.error() << " for arg " << e.argId());
    }

    LOG_E("Program exited with errors");
    return Code::ERROR;
}