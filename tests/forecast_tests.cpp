#include "fixtures/forecast.hpp"
#include "stealth/forecaster.hpp"
#include "test_support.hpp"
#include <algorithm>
#include <array>
#include <limits>
#include <string_view>
#include <vector>

using namespace stealth::numeric;
namespace f = forecast_fixture;
namespace {
ForecastConfig config() {
    return {f::features, f::width, f::heads, f::hidden, f::tokens, f::horizon, f::target};
}
void compare(std::span<const float> actual, std::span<const float> expected) {
    CHECK(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
        test::near(actual[i], expected[i], 5e-5, 2e-4);
}
void parity() {
    CHECK(forecast_parameter_count({7, 64, 4, 256, 96, 4, 6}) == 50703);
    for (auto kernel : {DenseKernel::reference, DenseKernel::packed}) {
        ForecastModel model(config(), f::parameters, kernel);
        ForecastWorkspace ws(config());
        std::array<float, f::tokens> out{};
        CHECK(model.forward(f::input, f::tokens, out, ws) == Error::ok);
        compare(out, f::expected);
        CHECK(model.config() == config());
        CHECK(model.parameter_bytes() > 0);
        CHECK(model.position_bytes() == f::tokens * f::width * sizeof(float));
        CHECK(ws.bytes() > 0);
    }
}
void causality() {
    ForecastModel model(config(), f::parameters);
    ForecastWorkspace ws(config()), other(config());
    std::array<float, f::tokens> full{}, out{}, prefix{};
    CHECK(model.forward(f::input, f::tokens, full, ws) == Error::ok);
    for (std::size_t n = 1; n <= f::tokens; ++n) {
        CHECK(model.forward(std::span(f::input).first(n * f::features), n,
                            std::span(prefix).first(n), other) == Error::ok);
        for (std::size_t i = 0; i < n; ++i)
            test::near(prefix[i], full[i], 0, 0);
        auto input = f::input;
        for (std::size_t i = n * f::features; i < input.size(); ++i)
            input[i] = static_cast<float>(i) * 3;
        CHECK(model.forward(input, f::tokens, out, ws) == Error::ok);
        for (std::size_t i = 0; i < n; ++i)
            test::near(out[i], full[i], 0, 0);
    }
}
void failures() {
    ForecastModel model(config(), f::parameters);
    ForecastWorkspace ws(config());
    std::array<float, f::tokens> out{};
    out.fill(123);
    CHECK(model.forward(f::input, f::tokens, out, ws) == Error::ok);
    out.fill(123);
    auto input = f::input;
    input.back() = std::numeric_limits<float>::infinity();
    CHECK(model.forward(input, f::tokens, out, ws) == Error::non_finite);
    test::unchanged(out);
    for (std::size_t n : {std::size_t{0}, f::tokens + 1, std::numeric_limits<std::size_t>::max()}) {
        CHECK(model.forward(f::input, n, out, ws) == Error::shape_mismatch);
        test::unchanged(out);
    }
    CHECK(model.forward(std::span(f::input).first(3), f::tokens, out, ws) == Error::shape_mismatch);
    test::unchanged(out);
    input = f::input;
    CHECK(model.forward(input, f::tokens, std::span(input).first(f::tokens), ws) ==
          Error::aliasing);
    CHECK(input == f::input);
    auto wrong = config();
    ++wrong.context;
    ForecastWorkspace wrong_ws(wrong);
    CHECK(model.forward(f::input, f::tokens, out, wrong_ws) == Error::shape_mismatch);
    test::unchanged(out);
    // Last-row feature normalization overflow must preserve every earlier output.
    auto p = f::parameters;
    p[f::features] = std::numeric_limits<float>::min();
    ForecastModel overflow(config(), p);
    input = f::input;
    for (std::size_t i = 0; i < f::tokens; ++i)
        input[i * f::features] = p[0];
    input[(f::tokens - 1) * f::features] = std::numeric_limits<float>::max();
    CHECK(overflow.forward(input, f::tokens, out, ws) == Error::numerical_error);
    test::unchanged(out);
    CHECK(model.forward(f::input, f::tokens, out, ws) == Error::ok);
    compare(out, f::expected);
    // First readout is finite; only the final readout overflows inverse scaling.
    ForecastConfig late_config{1, 2, 1, 1, 2, 4, 0};
    std::vector<float> late_p(forecast_parameter_count(late_config), 0);
    late_p[1] = std::numeric_limits<float>::max();
    late_p[2] = 1;
    late_p[late_p.size() - 7] = 2;
    late_p[late_p.size() - 3] = 1;
    ForecastModel late(late_config, late_p);
    ForecastWorkspace late_ws(late_config);
    std::array<float, 2> late_input{late_p[1], late_p[1]}, late_out{123, 123};
    CHECK(late.forward(std::span(late_input).first(1), 1, std::span(late_out).first(1), late_ws) ==
          Error::ok);
    test::near(late_out[0], 0, 0, 0);
    late_out.fill(123);
    CHECK(late.forward(late_input, 2, late_out, late_ws) == Error::numerical_error);
    test::unchanged(late_out);
    out.fill(123);
    auto moved = std::move(model);
    CHECK(model.forward(f::input, f::tokens, out, ws) == Error::invalid_state);
    test::unchanged(out);
    auto moved_ws = std::move(ws);
    CHECK(moved.forward(f::input, f::tokens, out, ws) == Error::invalid_state);
    test::unchanged(out);
    CHECK(moved.forward(f::input, f::tokens, out, moved_ws) == Error::ok);
    compare(out, f::expected);
}
void replay() {
    ForecastModel model(config(), f::parameters);
    ForecastWorkspace first(config()), second(config());
    std::array<float, f::tokens> out{};
    CHECK(model.forward(f::input, f::tokens, out, first) == Error::ok);
    for (int i = 0; i < 2000; ++i) {
        auto &ws = i % 2 == 0 ? first : second;
        CHECK(model.forward(f::input, f::tokens, out, ws) == Error::ok);
        compare(out, f::expected);
    }
}
void generated() {
    ForecastConfig c{7, 64, 4, 256, 96, 4, 6};
    const std::array<std::size_t, 20> sizes{
        7,  7,  64 * 7,   64,  64,       64, 3 * 64 * 64, 3 * 64, 64 * 64, 64,
        64, 64, 256 * 64, 256, 64 * 256, 64, 64,          64,     64,      1};
    std::vector<float> p;
    for (std::size_t segment = 0; segment < sizes.size(); ++segment)
        for (std::size_t i = 0; i < sizes[segment]; ++i) {
            float value;
            if (segment == 0)
                value = static_cast<float>(i) - 3;
            else if (segment == 1)
                value = static_cast<float>(i) + 1;
            else {
                value = static_cast<float>(static_cast<int>((i * 13 + segment * 7 + 3) % 29) - 14) /
                        128.0F;
                if (segment == 4 || segment == 10 || segment == 16)
                    value += 1;
            }
            p.push_back(value);
        }
    CHECK(p.size() == forecast_parameter_count(c));
    std::array<float, 96 * 7> input{};
    for (std::size_t i = 0; i < input.size(); ++i)
        input[i] = static_cast<float>(static_cast<int>((i * 13 + 17) % 29) - 14) / 8.0F;
    std::array<float, 96> reference{}, packed{};
    ForecastModel scalar(c, p, DenseKernel::reference), vector(c, p, DenseKernel::packed);
    ForecastWorkspace ws(c);
    CHECK(scalar.forward(input, 96, reference, ws) == Error::ok);
    CHECK(vector.forward(input, 96, packed, ws) == Error::ok);
    compare(reference, f::full_expected);
    compare(packed, f::full_expected);
    double worst = 0;
    std::size_t worst_origin = 0;
    for (std::size_t i = 0; i < 96; ++i) {
        test::near(packed[i], reference[i], 0, 0);
        const auto error = std::abs(static_cast<double>(packed[i]) - f::full_expected[i]);
        if (error > worst) {
            worst = error;
            worst_origin = i;
        }
    }
    std::cout << "Full task-shape max absolute error: " << worst << " at origin " << worst_origin
              << "\n";
    std::cout << "Full task-shape packed buffers: parameters=" << vector.parameter_bytes()
              << " positions=" << vector.position_bytes() << " workspace=" << ws.bytes()
              << " bytes\n";
}
void shapes() {
    // A constant scalar head has an exact physical-scale answer over odd and even widths.
    for (auto w : {std::size_t{1}, std::size_t{3}, std::size_t{8}, std::size_t{64}}) {
        ForecastConfig c{7, w, 1, 7, 96, 4, 6};
        auto count = forecast_parameter_count(c);
        std::vector<float> p(count, 0);
        std::fill_n(p.begin() + 7, 7, 2.0F);
        p[6] = 20;
        p.back() = 3;
        ForecastModel model(c, p);
        ForecastWorkspace ws(c);
        std::vector<float> input(96 * 7, 1), out(96, 123);
        CHECK(model.forward(input, 96, out, ws) == Error::ok);
        for (float y : out)
            test::near(y, 26, 0, 0);
    }
    for (ForecastConfig c : {ForecastConfig{64, 3, 1, 7, 1024, 1024, 63},
                             ForecastConfig{64, 512, 512, 2048, 1, 1, 63}}) {
        std::vector<float> p(forecast_parameter_count(c), 0);
        std::fill_n(p.begin() + static_cast<std::ptrdiff_t>(c.features), c.features, 2.0F);
        p[c.target] = 20;
        p.back() = 3;
        ForecastModel model(c, p);
        ForecastWorkspace ws(c);
        std::vector<float> input(c.context * c.features, 1), out(c.context, 123);
        CHECK(model.forward(input, c.context, out, ws) == Error::ok);
        for (float value : out)
            test::near(value, 26, 0, 0);
    }
    for (int which = 0; which < 9; ++which) {
        auto c = config();
        switch (which) {
        case 0:
            c.features = 0;
            break;
        case 1:
            c.features = max_features + 1;
            break;
        case 2:
            c.heads = 3;
            break;
        case 3:
            c.hidden = max_hidden + 1;
            break;
        case 4:
            c.context = max_context + 1;
            break;
        case 5:
            c.target = c.features;
            break;
        case 6:
            c.horizon = 0;
            break;
        case 7:
            c.epsilon = 0;
            break;
        default:
            c.width = std::numeric_limits<std::size_t>::max();
            break;
        }
        test::throws<std::logic_error>([&] { ForecastModel bad(c, f::parameters); });
        test::throws<std::logic_error>([&] { ForecastWorkspace bad(c); });
    }
    test::throws<std::invalid_argument>(
        [] { ForecastModel bad(config(), std::span(f::parameters).first(1)); });
    for (float value : {0.0F, -1.0F, std::numeric_limits<float>::quiet_NaN()}) {
        auto p = f::parameters;
        p[f::features] = value;
        test::throws<std::invalid_argument>([&] { ForecastModel bad(config(), p); });
    }
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc != 2)
            throw std::runtime_error("Expected test group");
        const std::string_view group = argv[1];
        if (group == "parity")
            parity();
        else if (group == "causality")
            causality();
        else if (group == "failures")
            failures();
        else if (group == "replay")
            replay();
        else if (group == "shapes")
            shapes();
        else if (group == "generated")
            generated();
        else
            throw std::runtime_error("Unknown group");
        std::cout << group << ": " << test::checks << " checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
