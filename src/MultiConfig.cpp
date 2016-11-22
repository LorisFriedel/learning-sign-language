//
// @author Loris Friedel
//

#include <cv.hpp>
#include "../inc/MultiConfig.hpp"
#include "../inc/log.h"

MultiConfig::MultiConfig(std::string configPath) throw(ParsingException) {
    using namespace cv;
    FileStorage fs(configPath, FileStorage::READ);

    if (fs.isOpened()) {
        fs["dataDir"] >> dataDir;
        fs["modelDir"] >> modelDir;
        fs["logDir"] >> logDir;

        FileNode fsNames = fs["names"];
        for (FileNodeIterator it = fsNames.begin(); it != fsNames.end(); ++it) {
            names.push_back(*it);
        }

        FileNode fsTypes = fs["types"];
        for (FileNodeIterator it = fsTypes.begin(); it != fsTypes.end(); ++it) {
            types.push_back(*it);
        }

        FileNode fsTopologies = fs["topologies"];
        for (FileNodeIterator it = fsTopologies.begin(); it != fsTopologies.end(); ++it) {
            topologies.push_back(*it);
        }

        fs.release();
    } else {
        throw ParsingException(configPath);;
    }
}