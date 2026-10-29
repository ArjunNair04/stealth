#include "fixtures/forecast.hpp"
#include "stealth/forecaster.hpp"
#include "test_support.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>
using namespace stealth::numeric;
namespace f = forecast_fixture;
namespace {
void same(float a, float b) {
    CHECK(std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b));
}
void attention() {
    test::Random rng;
    for (std::size_t width : {1U, 4U, 8U, 12U, 64U})
        for (std::size_t heads : {1U, 2U, 4U}) {
            if (width % heads != 0)
                continue;
            for (auto kernel : {AttentionKernel::reference, AttentionKernel::tiled}) {
                constexpr std::size_t tokens = 17;
                AttentionWorkspace ws(tokens, width, heads, kernel);
                std::vector<float> q(tokens * width), k(q.size()), v(q.size()), all(q.size()),
                    last(width);
                for (auto *values : {&q, &k, &v})
                    for (auto &x : *values)
                        x = rng.next();
                CHECK(causal_attention(q, k, v, tokens, all, ws) == Error::ok);
                for (std::size_t n = 1; n <= tokens; ++n) {
                    CHECK(causal_attention(std::span(q).first(n * width),
                                           std::span(k).first(n * width),
                                           std::span(v).first(n * width), n, last, ws,
                                           SequenceOutput::last) == Error::ok);
                    for (std::size_t i = 0; i < width; ++i)
                        same(last[i], all[(n - 1) * width + i]);
                }
                std::fill(last.begin(), last.end(), 123.0F);
                CHECK(causal_attention(q, k, v, tokens, last, ws,
                                       static_cast<SequenceOutput>(99)) == Error::shape_mismatch);
                CHECK(causal_attention(q, k, v, 0, last, ws, SequenceOutput::last) ==
                      Error::shape_mismatch);
                q.front() = std::numeric_limits<float>::quiet_NaN();
                CHECK(causal_attention(q, k, v, tokens, last, ws, SequenceOutput::last) ==
                      Error::non_finite);
                test::unchanged(last);
            }
        }
}
void forecast() {
    ForecastConfig c{f::features, f::width, f::heads, f::hidden, f::tokens, f::horizon, f::target};
    for (auto kernel : {DenseKernel::reference, DenseKernel::packed}) {
        ForecastModel m(c, f::parameters, kernel);
        ForecastWorkspace ws(c);
        std::array<float, f::tokens> all{};
        std::array<float, 1> last{};
        CHECK(m.forward(f::input, f::tokens, all, ws) == Error::ok);
        for (std::size_t n = 1; n <= f::tokens; ++n) {
            CHECK(m.forward(std::span(f::input).first(n * f::features), n, last, ws,
                            SequenceOutput::last) == Error::ok);
            same(last[0], all[n - 1]);
            test::near(last[0], f::expected[n - 1], 5e-5, 2e-4);
        }
        CHECK(m.forward(f::input, f::tokens, all, ws) == Error::ok);
        same(last[0], all.back());
    }
    test::Random rng;
    for (auto mode : {ForecastOutput::level, ForecastOutput::correction})
        for (std::size_t width : {4U, 12U, 64U}) {
            ForecastConfig shape{3, width, 4, 2 * width + 1, 17, 1, 2, 1e-5F, mode};
            std::vector<float> p(forecast_parameter_count(shape)), x(17 * 3), all(17);
            for (auto &v : p)
                v = rng.next() / 8;
            std::fill(p.begin() + 3, p.begin() + 6, 1.0F);
            for (auto &v : x)
                v = rng.next();
            for (auto kernel : {DenseKernel::reference, DenseKernel::packed}) {
                ForecastModel m(shape, p, kernel);
                ForecastWorkspace ws(shape);
                std::array<float, 1> last{};
                CHECK(m.forward(x, 17, all, ws) == Error::ok);
                for (std::size_t n = 1; n <= 17; ++n) {
                    CHECK(m.forward(std::span(x).first(n * 3), n, last, ws, SequenceOutput::last) ==
                          Error::ok);
                    same(last[0], all[n - 1]);
                }
            }
        }
}
void failures() {
    ForecastConfig c{f::features, f::width, f::heads, f::hidden, f::tokens, f::horizon, f::target};
    ForecastModel m(c, f::parameters);
    ForecastWorkspace ws(c);
    std::array<float, 1> last{123};
    CHECK(m.forward(f::input, f::tokens, last, ws, static_cast<SequenceOutput>(99)) ==
          Error::shape_mismatch);
    CHECK(m.forward(f::input, 0, last, ws, SequenceOutput::last) == Error::shape_mismatch);
    CHECK(m.forward(f::input, f::tokens, {}, ws, SequenceOutput::last) == Error::shape_mismatch);
    auto x = f::input;
    const auto original = x;
    CHECK(m.forward(x, f::tokens, std::span(x).first(1), ws, SequenceOutput::last) ==
          Error::aliasing);
    CHECK(x == original);
    x.front() = std::numeric_limits<float>::infinity();
    CHECK(m.forward(x, f::tokens, last, ws, SequenceOutput::last) == Error::non_finite);
    test::unchanged(last);
    auto p = f::parameters;
    p[c.features] = std::numeric_limits<float>::min();
    ForecastModel overflow(c, p);
    x = f::input;
    x.front() = std::numeric_limits<float>::max();
    CHECK(overflow.forward(x, f::tokens, last, ws, SequenceOutput::last) == Error::numerical_error);
    test::unchanged(last);
    auto moved = std::move(ws);
    CHECK(m.forward(f::input, f::tokens, last, ws, SequenceOutput::last) == Error::invalid_state);
    CHECK(m.forward(f::input, f::tokens, last, moved, SequenceOutput::last) == Error::ok);
    // An unused earlier MLP overflow is deliberately outside the latest-only DAG.
    std::array<float, 2> zero{}, one{1, 1}, huge{};
    huge.fill(std::numeric_limits<float>::max() * 0.5F);
    std::array<float, 12> qw{};
    std::array<float, 6> qb{};
    std::array<float, 4> pw{};
    std::array<float, 1> up_bias{1};
    TransformerBlock block(2, 1, 1,
                           {one, zero, qw, qb, pw, zero, {zero, zero, zero, up_bias, huge, zero}});
    TransformerWorkspace block_ws(2, 2, 1, 1);
    std::array<float, 4> input{std::numeric_limits<float>::max(), 0, 0, 0}, full{};
    full.fill(123.0F);
    std::array<float, 2> point{};
    CHECK(block.forward(input, 2, full, block_ws) == Error::numerical_error);
    test::unchanged(full);
    CHECK(block.forward(input, 2, point, block_ws, SequenceOutput::last) == Error::ok);
    const auto activation = static_cast<float>(0.5 * std::erfc(-0.7071067811865475244));
    for (float value : point)
        test::near(value, static_cast<float>(activation * huge[0]), 0, 0);
    point.fill(123.0F);
    CHECK(block.forward(input, 2, point, block_ws, static_cast<SequenceOutput>(99)) ==
          Error::shape_mismatch);
    test::unchanged(point);
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc != 2)
            throw std::runtime_error("Expected group");
        const std::string_view group(argv[1]);
        if (group == "attention")
            attention();
        else if (group == "forecast")
            forecast();
        else if (group == "failures")
            failures();
        else
            throw std::runtime_error("Unknown group");
        std::cout << group << ": " << test::checks << " checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
