//
// @author Loris Friedel
//

#pragma once

#include <iostream>

/* Log */

#define LOG_I(str) std::cout << str << std::endl
#define LOGP_I(prefix, str) std::cout << prefix << "_::" << str << std::endl;
#define LOG_E(str) std::cerr << str << std::endl
#define LOGP_E(prefix, str) std::cerr << prefix << "_::" << str << std::endl;
