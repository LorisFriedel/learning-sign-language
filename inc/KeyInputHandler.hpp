//
// @author Loris Friedel
//

#pragma once

#include <functional>
#include <map>
#include <vector>

class KeyInputHandler {
public:
    void bind(int key, std::function<void(const int &)> *function);

    void unbind(int key);

    void apply(int key);

private:
    std::map<int, std::vector<std::function<void(const int &)> *>> m_key_binding;
};