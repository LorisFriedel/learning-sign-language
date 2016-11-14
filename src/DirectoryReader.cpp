//
// @author Loris Friedel
//

#include <dirent.h>
#include <opencv2/core/mat.hpp>
#include <sstream>
#include "../inc/DirectoryReader.hpp"
#include "../inc/code.h"

DirectoryReader::DirectoryReader(std::string directory)
        : directory(directory) {}

int DirectoryReader::foreachFile(const std::function<void(std::string, std::string)> &callback) {
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(directory.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            if (ent->d_type == DT_REG) {
                std::string fileName = ent->d_name;

                std::stringstream full_path;
                full_path << directory << "/" << fileName;

                callback(full_path.str(), fileName);
            }
        }
        closedir(dir);
        return Code::SUCCESS;
    } else {
        // Could not open directory
        return Code::ERROR;
    }
}
