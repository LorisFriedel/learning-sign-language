//
// @author Loris Friedel
//

#pragma once

#include <string>
#include <vector>

class MultiConfig {
public:
    class JsonParsingException : public std::exception {
    public:
        JsonParsingException(std::string filePath) : filePath(filePath) {}

        const char *what() const throw() {
            std::string errorTxt = "Error while parsing configuration file: " + filePath;
            return errorTxt.c_str();
        }

        std::string filePath;
    };

    MultiConfig(std::string configPath) throw(JsonParsingException);

    std::string trainDir;
    std::string testDir;
    std::vector<std::string> names;
    std::vector<std::string> suffixes;
    std::vector<std::string> topologies;
};

