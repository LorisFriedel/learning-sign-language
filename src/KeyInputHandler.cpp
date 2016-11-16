//
// @author Loris Friedel
//

#include "../inc/KeyInputHandler.hpp"

void KeyInputHandler::bind(int key, std::function<void(const int &)> *function) {
    keyBinding[key].push_back(function);
}

void KeyInputHandler::unbind(int key) {
    keyBinding.erase(key);
}

void KeyInputHandler::apply(int key) {
    if (keyBinding.count(key) > 0) {
        for (std::function<void(const int &)> *function : keyBinding[key]) {
            if (function != nullptr) {
                (*function)(key);
            }
        }
    }
}
