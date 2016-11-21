//
// @author Loris Friedel
//

#include <cv.hpp>
#include "../inc/MultiConfig.hpp"
#include "../inc/log.h"

MultiConfig::MultiConfig(std::string configPath) throw(JsonParsingException) {
    using namespace cv;
    FileStorage fs(configPath, FileStorage::READ);

    if (fs.isOpened()) {
        fs["trainDir"] >> trainDir;
        fs["testDir"] >> testDir;

        FileNode fsNames = fs["names"];
        for (FileNodeIterator it = fsNames.begin(); it != fsNames.end(); ++it) {
            names.push_back(*it);
        }

        FileNode fsSuffixes = fs["suffixes"];
        for (FileNodeIterator it = fsSuffixes.begin(); it != fsSuffixes.end(); ++it) {
            suffixes.push_back(*it);
        }

        FileNode fsTopologies = fs["topologies"];
        for (FileNodeIterator it = fsTopologies.begin(); it != fsTopologies.end(); ++it) {
            topologies.push_back(*it);
        }

        fs.release();
    } else {
        throw JsonParsingException(configPath);;
    }
}