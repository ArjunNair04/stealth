#include "../tests/fixtures/forecast.hpp"
#include "stealth/model_file.hpp"
#include <array>
#include <cmath>
#include <iostream>

int main() {
    using namespace stealth::numeric;
    namespace f = forecast_fixture;
    try {
        auto model = load_forecaster(std::as_bytes(std::span(f::artifact)));
        ForecastWorkspace workspace(model->config());
        std::array<float, f::tokens> output{};
        if (model->forward(f::input, f::tokens, output, workspace) != Error::ok)
            return 1;
        for (std::size_t i = 0; i < output.size(); ++i)
            if (!std::isfinite(output[i]) ||
                std::abs(output[i] - f::expected[i]) > 5e-5F + 2e-4F * std::abs(f::expected[i]))
                return 1;
        std::cout << "Diagnostic only: invented readings and untrained weights.\n"
                     "Loaded one block; scaled 7 inputs per reading; added positions; produced one "
                     "point forecast.\n"
                  << "Final diagnostic output: " << output.back() << "\n"
                  << "Independent equation check: passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
