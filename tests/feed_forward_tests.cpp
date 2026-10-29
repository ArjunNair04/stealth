#include "fixtures/feed_forward.hpp"
#include "stealth/feed_forward.hpp"
#include "test_support.hpp"
#include <algorithm>
#include <array>
#include <limits>
#include <string_view>
#include <vector>

using namespace stealth::numeric;
namespace {
FeedForwardParameters parameters() {
    return {fixture::scale,   fixture::shift,        fixture::up_weights,
            fixture::up_bias, fixture::down_weights, fixture::down_bias};
}
void parity() {
    for (auto kernel : {DenseKernel::reference, DenseKernel::packed}) {
        FeedForward model(fixture::width, fixture::hidden, parameters(), fixture::epsilon, kernel);
        FeedForwardWorkspace workspace(fixture::width, fixture::hidden);
        std::array<float, fixture::width> output{};
        const std::array inputs{fixture::input, fixture::input_other, fixture::input_constant};
        const std::array expected{fixture::expected, fixture::expected_other,
                                  fixture::expected_constant};
        for (std::size_t n = 0; n < inputs.size(); ++n) {
            CHECK(model.forward(inputs[n], output, workspace) == Error::ok);
            for (std::size_t i = 0; i < output.size(); ++i)
                test::near(output[i], expected[n][i], 2e-5, 5e-5);
        }
        CHECK(workspace.bytes() == (2 * fixture::width + 3 * fixture::hidden) * sizeof(float));
        CHECK(model.parameter_bytes() > 0);
    }
}
void failures() {
    FeedForward model(fixture::width, fixture::hidden, parameters());
    FeedForwardWorkspace workspace(fixture::width, fixture::hidden), wrong(3, 8);
    std::array<float, fixture::width> output{123, 123, 123, 123};
    CHECK(model.forward(fixture::input, output, wrong) == Error::shape_mismatch);
    test::unchanged(output);
    CHECK(model.forward(std::span(fixture::input).first(3), output, workspace) ==
          Error::shape_mismatch);
    test::unchanged(output);
    auto input = fixture::input;
    input[0] = std::numeric_limits<float>::quiet_NaN();
    CHECK(model.forward(input, output, workspace) == Error::non_finite);
    test::unchanged(output);
    input = fixture::input;
    CHECK(model.forward(input, input, workspace) == Error::aliasing);
    CHECK(input == fixture::input);
    auto bad = parameters();
    bad.scale = std::span(fixture::scale).first(3);
    test::throws<std::invalid_argument>([&] { FeedForward invalid(4, 7, bad); });
    test::throws<std::invalid_argument>([&] { FeedForward invalid(4, 7, parameters(), 0); });
    test::throws<std::invalid_argument>([] { FeedForwardWorkspace invalid(0, 7); });
    test::throws<std::length_error>(
        [] { FeedForwardWorkspace invalid(max_elements, max_elements); });
    // Fail late in the down projection, after internal workspace has changed.
    std::array<float, 4> scale{1, 1, 1, 1}, shift{1, 1, 1, 1};
    std::array<float, 4> zero{}, huge{};
    huge.fill(std::numeric_limits<float>::max());
    std::array<float, 1> up_bias{2};
    FeedForward bad_model(4, 1, {scale, shift, zero, up_bias, huge, zero});
    FeedForwardWorkspace bad_workspace(4, 1);
    CHECK(bad_model.forward(fixture::input, output, bad_workspace) == Error::numerical_error);
    test::unchanged(output);
    // A failed call cannot poison later calls with the same workspace.
    CHECK(model.forward(fixture::input, output, workspace) == Error::ok);
    for (std::size_t i = 0; i < output.size(); ++i)
        test::near(output[i], fixture::expected[i], 2e-5, 5e-5);
    auto moved_workspace = std::move(workspace);
    CHECK(model.forward(fixture::input, output, workspace) == Error::invalid_state);
    CHECK(model.forward(fixture::input, output, moved_workspace) == Error::ok);
    auto moved_model = std::move(model);
    CHECK(model.forward(fixture::input, output, moved_workspace) == Error::invalid_state);
    CHECK(moved_model.forward(fixture::input, output, moved_workspace) == Error::ok);
}
void replay() {
    FeedForward model(fixture::width, fixture::hidden, parameters());
    FeedForwardWorkspace first(fixture::width, fixture::hidden),
        second(fixture::width, fixture::hidden);
    std::array<float, fixture::width> a{}, b{};
    for (int trial = 0; trial < 10000; ++trial) {
        CHECK(model.forward(fixture::input, a, first) == Error::ok);
        CHECK(model.forward(fixture::input_other, b, second) == Error::ok);
        for (std::size_t i = 0; i < a.size(); ++i) {
            test::near(a[i], fixture::expected[i], 2e-5, 5e-5);
            test::near(b[i], fixture::expected_other[i], 2e-5, 5e-5);
        }
    }
}
void generated() {
    test::Random random;
    for (std::size_t width : {1U, 3U, 8U, 17U, 64U}) {
        for (std::size_t hidden : {1U, 7U, 16U, 33U}) {
            std::vector<float> scale(width, 1), shift(width), w1(width * hidden), b1(hidden),
                w2(width * hidden), b2(width), x(width), a(width), b(width);
            for (auto &v : w1)
                v = random.next() / 8;
            for (auto &v : w2)
                v = random.next() / 8;
            for (auto &v : b1)
                v = random.next() / 8;
            for (auto &v : b2)
                v = random.next() / 8;
            for (auto &v : x)
                v = random.next();
            FeedForwardParameters p{scale, shift, w1, b1, w2, b2};
            FeedForward reference(width, hidden, p, 1e-5F, DenseKernel::reference),
                packed(width, hidden, p);
            FeedForwardWorkspace wa(width, hidden), wb(width, hidden);
            CHECK(reference.forward(x, a, wa) == Error::ok);
            CHECK(packed.forward(x, b, wb) == Error::ok);
            for (std::size_t i = 0; i < width; ++i)
                test::near(a[i], b[i], 2e-5, 5e-5);
        }
    }
}
} // namespace
int main(int argc, char *argv[]) {
    try {
        if (argc != 2)
            throw std::runtime_error("Expected group");
        const std::string_view group(argv[1]);
        if (group == "parity")
            parity();
        else if (group == "failures")
            failures();
        else if (group == "replay")
            replay();
        else if (group == "generated")
            generated();
        else
            throw std::runtime_error("Unknown group");
        std::cout << group << ": " << test::checks << " checks passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
