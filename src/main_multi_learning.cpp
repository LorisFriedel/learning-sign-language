//
// @author Loris Friedel
//

#include <tclap/CmdLine.h>
#include <cv.hpp>
#include <dirent.h>
#include <regex>
#include <random>
#include "../inc/code.h"
#include "../inc/log.h"
#include "../inc/constant.h"
#include "../inc/MLPHand.hpp"
#include "../inc/time.h"
#include "../inc/DataYmlReader.hpp"
#include "../inc/DirectoryReader.hpp"
#include "../inc/Timer.hpp"
#include "../inc/MultiConfig.hpp"

int main(int argc, const char **argv) {
    try {
        TCLAP::CmdLine cmd(
                "!!! Help for sign language multi-threaded learning program. !!!"
                        "\nWritten by Loris Friedel",
                ' ', "1.0");

        TCLAP::ValueArg<std::string> configFileArg("c", "config", "Configuration file. See multi_learning_config_example.yml for details.", true, "", "PATH_TO_JSON_CONFIG_FILE", cmd);


        //// Parse the argv array
        cmd.parse(argc, argv);

        MultiConfig config("./multi_learning_config_example.yml");

        //// Get the value parsed by each arg and handle them
        int a = 0;
        //std::string testDir = testDirArg.getValue();
        return Code::SUCCESS;
    } catch (TCLAP::ArgException &e) {  // catch any exceptions
        LOG_E("error: " << e.error() << " for arg " << e.argId());
    }

    LOG_E("Program exited with errors");
    return Code::ERROR;
}