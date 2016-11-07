//
// @author Loris Friedel
//

#include "../inc/KeyInputHandler.hpp"

void KeyInputHandler::bind(int key, std::function<void(const int &)> *function) {
    m_key_binding[key].push_back(function);
}

void KeyInputHandler::unbind(int key) {
    m_key_binding.erase(key);
}

void KeyInputHandler::apply(int key) {
    if (m_key_binding.count(key) > 0) {
        for (std::function<void(const int &)> *function : m_key_binding[key]) {
            if (function != nullptr) {
                (*function)(key);
            }
        }
    }
}
