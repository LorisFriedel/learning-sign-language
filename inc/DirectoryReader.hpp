//
// @author Loris Friedel
//

#pragma once

#include <string>
#include <functional>

class DirectoryReader {
public:
    DirectoryReader(std::string directory);

    int foreachFile(const std::function<void(std::string, std::string)> &callback);

private:
    std::string directory;
};

