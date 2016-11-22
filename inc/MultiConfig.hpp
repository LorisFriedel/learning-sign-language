//
// @author Loris Friedel
//

#pragma once

#include <string>
#include <vector>

class MultiConfig {
public:
    class ParsingException : public std::exception {
    public:
        ParsingException(std::string filePath) : filePath(filePath) {}

        const char *what() const throw() {
            std::string errorTxt = "Error while parsing configuration file: " + filePath;
            return errorTxt.c_str();
        }

        std::string filePath;
    };

    MultiConfig(std::string configPath) throw(ParsingException);

    std::string dataDir;
    std::string modelDir;
    std::vector<std::string> names;
    std::vector<std::string> types;
    std::vector<std::string> topologies;
};

