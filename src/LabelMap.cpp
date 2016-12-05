/**
 * @author Loris Friedel
 */


#include "../inc/LabelMap.hpp"

void LabelMap::read(const cv::FileNode &node) {
    labelMap.clear();
    std::for_each(node.begin(), node.end(), [this](const cv::FileNode &n) {
        this->labelMap[n] = n.name();
    });
}

void LabelMap::write(cv::FileStorage &fs) const {
    fs << "{";
    for (auto it = labelMap.begin(); it != labelMap.end(); ++it) {
        fs << it->second << it->first;
    }
    fs << "}";
}

void LabelMap::put(int key, std::string value) {
    labelMap[key] = value;
}

void LabelMap::clear() {
    labelMap.clear();
}

std::string LabelMap::get(int key) {
    if(labelMap.find(key) != labelMap.end()) {
        return labelMap[key];
    }
    return std::to_string(key);
}

void write(cv::FileStorage &fs, const std::string &, const LabelMap &obj) {
    obj.write(fs);
}

void read(const cv::FileNode &node, LabelMap &obj, const LabelMap &default_value) {
    if (node.empty()) {
        obj = default_value;
    } else {
        obj.read(node);
    }
}