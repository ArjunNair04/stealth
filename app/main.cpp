#include "stealth/model_file.hpp"
#include "stealth/runtime.hpp"

#include <iostream>
#include <string_view>

namespace {
void usage(std::ostream &output) {
    output << "Usage: stealth [--version | --build-info | --inspect-model FILE | --help]\n"
           << "Native CPU forecaster; see the recorded correctness, accuracy and runtime results.\n";
}
} // namespace

int main(int argc, char *argv[]) {
    if (argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--help")) {
        usage(std::cout);
        return 0;
    }
    if (argc == 3 && std::string_view(argv[1]) == "--inspect-model") {
        try {
            auto model = stealth::numeric::load_forecaster_file(argv[2]);
            const auto c = model->config();
            stealth::numeric::ForecastWorkspace workspace(c);
            const bool correction = c.output == stealth::numeric::ForecastOutput::correction;
            std::cout << "format_version: " << (correction ? 2 : 1)
                      << "\noutput_mode: " << (correction ? "correction" : "level")
                      << "\nblocks: 1\nfeatures: " << c.features
                      << "\nwidth: " << c.width << "\nheads: " << c.heads
                      << "\nhidden: " << c.hidden << "\ncontext: " << c.context
                      << "\nhorizon_steps: " << c.horizon << "\ntarget_feature: " << c.target
                      << "\nparameter_bytes: " << model->parameter_bytes()
                      << "\nposition_bytes: " << model->position_bytes()
                      << "\nworkspace_bytes: " << workspace.bytes() << "\n";
            return 0;
        } catch (const std::exception &e) {
            std::cerr << "Model rejected: " << e.what() << '\n';
            return 1;
        }
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
                  << "source_fingerprint: " << info.source_fingerprint << '\n'
                  << "cxx_standard: 20\n";
    } else {
        std::cerr << "Unknown argument: " << argument << '\n';
        usage(std::cerr);
        return 2;
    }
    return 0;
}
