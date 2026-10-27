#include "stealth/runtime.hpp"

#include <iostream>
#include <string_view>

namespace {
void usage(std::ostream &output) {
    output << "Usage: stealth [--version | --build-info | --help]\n"
           << "Native development foundation; forecasting is not implemented yet.\n";
}
} // namespace

int main(int argc, char *argv[]) {
    if (argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--help")) {
        usage(std::cout);
        return 0;
    }
    if (argc != 2) {
        usage(std::cerr);
        return 2;
    }
    const auto info = stealth::build_info();
    const std::string_view argument(argv[1]);
    if (argument == "--version") {
        std::cout << "stealth " << info.version << '\n';
    } else if (argument == "--build-info") {
        std::cout << "version: " << info.version << '\n'
                  << "compiler: " << info.compiler << '\n'
                  << "operating_system: " << info.operating_system << '\n'
                  << "architecture: " << info.architecture << '\n'
                  << "configuration: " << info.configuration << '\n'
                  << "cxx_standard: 20\n";
    } else {
        std::cerr << "Unknown argument: " << argument << '\n';
        usage(std::cerr);
        return 2;
    }
    return 0;
}
