/**
 * @author Loris Friedel
 */

#pragma once

#include <cv.hpp>
#include <map>

class LabelMap {
public:
    void put(int key, std::string value);

    std::string get(int key);

    void clear();

    void write(cv::FileStorage &fs) const;

    void read(const cv::FileNode &node);

private:
    std::map<int, std::string> labelMap;
};

void write(cv::FileStorage &fs, const std::string &, const LabelMap &obj);

void read(const cv::FileNode &node, LabelMap &obj, const LabelMap &default_value = LabelMap());

