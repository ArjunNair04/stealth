#include "stealth/runtime.hpp"

#include <array>
#include <iostream>
#include <span>

int main() {
    // Exercise a C++20 standard-library header as well as the linked native library.
    // MSVC reports its real language level through _MSVC_LANG.
#ifdef _MSVC_LANG
    static_assert(_MSVC_LANG >= 202002L);
#else
    static_assert(__cplusplus >= 202002L);
#endif
    const auto info = stealth::build_info();
    const std::array fields{info.version, info.compiler, info.operating_system, info.architecture,
                            info.configuration};
    for (const auto field : std::span(fields)) {
        if (field.empty()) {
            std::cerr << "The linked library returned an empty build field\n";
            return 1;
        }
    }
    std::cout << "C++20 consumer linked successfully: " << info.version << '\n';
    return 0;
}
