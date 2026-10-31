#include "fixtures/attention.hpp"
#include "fixtures/forecast.hpp"
#include "stealth/forecaster.hpp"
#include "test_support.hpp"
#include <algorithm>
#include <bit>
#include <limits>
#include <string_view>
#include <vector>
using namespace stealth::numeric;
namespace f = forecast_fixture;
namespace {
ForecastConfig config() {
    return {f::features, f::width, f::heads, f::hidden, f::tokens, f::horizon, f::target};
}
void exact(std::span<const float> a, std::span<const float> b) {
    CHECK(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i)
        CHECK(std::bit_cast<std::uint32_t>(a[i]) == std::bit_cast<std::uint32_t>(b[i]));
}
void parity() {
    for (auto kernel : {DenseKernel::reference, DenseKernel::packed})
        for (auto mode : {SequenceOutput::last, SequenceOutput::all}) {
            ForecastModel model(config(), f::parameters, kernel);
            ForecastWorkspace serial(config());
            ForecastBatchWorkspace ws(config(), 17);
            for (std::size_t batch : {1U, 2U, 3U, 4U, 5U, 8U, 17U})
                for (std::size_t tokens : {1U, 3U, static_cast<unsigned>(f::tokens)}) {
                    const auto out_tokens = mode == SequenceOutput::last ? 1 : tokens;
                    std::vector<float> input(batch * tokens * f::features), out(batch * out_tokens),
                        expected(out.size());
                    for (std::size_t b = 0; b < batch; ++b)
                        for (std::size_t i = 0; i < tokens * f::features; ++i)
                            input[b * tokens * f::features + i] =
                                f::input[i] + static_cast<float>(b) / 16;
                    for (std::size_t b = 0; b < batch; ++b)
                        CHECK(model.forward(std::span(input).subspan(b * tokens * f::features,
                                                                     tokens * f::features),
                                            tokens,
                                            std::span(expected).subspan(b * out_tokens, out_tokens),
                                            serial, mode) == Error::ok);
                    CHECK(model.forward_batch(input, batch, tokens, out, ws, mode) == Error::ok);
                    exact(out, expected);
                    // Independently generated Torch fixture for the first unmodified window.
                    test::near(out[0], f::expected[mode == SequenceOutput::last ? tokens - 1 : 0],
                               5e-5, 2e-4);
                    // Poison workspace with other valid work, then replay the original batch.
                    std::vector<float> changed(input.size(), 1.25F), scratch(out.size());
                    CHECK(model.forward_batch(changed, batch, tokens, scratch, ws, mode) ==
                          Error::ok);
                    CHECK(model.forward_batch(input, batch, tokens, out, ws, mode) == Error::ok);
                    exact(out, expected);
                }
        }
}
void block() {
    namespace a = attention_fixture;
    TransformerBlock model(
        a::width, a::heads, a::hidden,
        {a::attn_scale,
         a::attn_shift,
         a::qkv_weights,
         a::qkv_bias,
         a::projection_weights,
         a::projection_bias,
         {a::ff_scale, a::ff_shift, a::up_weights, a::up_bias, a::down_weights, a::down_bias}},
        a::epsilon);
    TransformerWorkspace ws(a::tokens, a::width, a::heads, a::hidden, 5);
    TransformerWorkspace serial(a::tokens, a::width, a::heads, a::hidden);
    std::vector<float> input(5 * a::input.size());
    for (std::size_t b = 0; b < 5; ++b)
        for (std::size_t i = 0; i < a::input.size(); ++i)
            input[b * a::input.size() + i] = a::input[i] + static_cast<float>(b * (i % 3)) / 7;
    for (auto mode : {SequenceOutput::all, SequenceOutput::last}) {
        const auto count = mode == SequenceOutput::last ? a::width : a::input.size();
        std::vector<float> out(5 * count, 123), expected(out.size());
        CHECK(model.forward_batch(input, 6, a::tokens, out, ws, mode) == Error::shape_mismatch);
        CHECK(model.forward_batch(input, 5, a::tokens, out, serial, mode) == Error::shape_mismatch);
        CHECK(model.forward_batch(input, 5, a::tokens + 1, out, ws, mode) == Error::shape_mismatch);
        CHECK(model.forward_batch(input, 5, a::tokens, out, ws, static_cast<SequenceOutput>(-1)) ==
              Error::shape_mismatch);
        CHECK(model.forward_batch(input, 5, a::tokens, std::span(input).first(out.size()), ws,
                                  mode) == Error::aliasing);
        const auto saved = input.back();
        input.back() = std::numeric_limits<float>::quiet_NaN();
        CHECK(model.forward_batch(input, 5, a::tokens, out, ws, mode) == Error::non_finite);
        input.back() = saved;
        test::unchanged(out);
        for (std::size_t b = 0; b < 5; ++b)
            CHECK(model.forward(std::span(input).subspan(b * a::input.size(), a::input.size()),
                                a::tokens, std::span(expected).subspan(b * count, count), serial,
                                mode) == Error::ok);
        CHECK(model.forward_batch(input, 5, a::tokens, out, ws, mode) == Error::ok);
        exact(out, expected);
        for (std::size_t i = 0; i < count; ++i)
            test::near(
                out[i],
                a::expected[(mode == SequenceOutput::last ? a::input.size() - a::width : 0) + i],
                3e-5, 1e-4);
    }
    for (auto batch :
         {std::size_t{0}, std::size_t{1025}, std::numeric_limits<std::size_t>::max()}) {
        bool caught = false;
        try {
            TransformerWorkspace bad(1, 2, 1, 2, batch);
        } catch (const std::exception &) {
            caught = true;
        }
        CHECK(caught);
    }
    bool caught = false;
    try {
        TransformerWorkspace bad(1024, 512, 8, 2048, 1024);
    } catch (const std::length_error &) {
        caught = true;
    }
    CHECK(caught);
}
void shapes() {
    for (std::size_t width : {3U, 8U, 16U, 33U, 64U, 128U}) {
        ForecastConfig c{3, width, width % 4 == 0 ? 4U : 1U, width * 2 + 1, 33, 1, 2};
        std::vector<float> p(forecast_parameter_count(c));
        for (std::size_t i = 0; i < p.size(); ++i)
            p[i] = static_cast<float>(static_cast<int>((i * 37 + 13) % 257) - 128) / 512;
        std::fill(p.begin() + 3, p.begin() + 6, 1.0F);
        for (auto kernel : {DenseKernel::reference, DenseKernel::packed}) {
            ForecastModel model(c, p, kernel);
            ForecastWorkspace serial(c);
            ForecastBatchWorkspace ws(c, 8);
            for (std::size_t batch : {2U, 5U, 8U})
                for (std::size_t tokens : {1U, 7U, 33U})
                    for (auto mode : {SequenceOutput::all, SequenceOutput::last}) {
                        std::vector<float> input(batch * tokens * c.features);
                        for (std::size_t i = 0; i < input.size(); ++i)
                            input[i] =
                                static_cast<float>(static_cast<int>((i * 11 + 5) % 37) - 18) / 8;
                        const auto output_tokens = mode == SequenceOutput::last ? 1 : tokens;
                        std::vector<float> out(batch * output_tokens), expected(out.size());
                        for (std::size_t b = 0; b < batch; ++b)
                            CHECK(model.forward(
                                      std::span(input).subspan(b * tokens * c.features,
                                                               tokens * c.features),
                                      tokens,
                                      std::span(expected).subspan(b * output_tokens, output_tokens),
                                      serial, mode) == Error::ok);
                        CHECK(model.forward_batch(input, batch, tokens, out, ws, mode) ==
                              Error::ok);
                        exact(out, expected);
                    }
        }
    }
    // Maximum configuration and batch capacity: scratch is bounded by four windows.
    ForecastBatchWorkspace bounded({64, 512, 8, 2048, 1024, 1, 63}, 1024);
    CHECK(bounded.bytes() < 80 * 1024 * 1024);
}
void failures() {
    const auto c = config();
    ForecastModel model(c, f::parameters);
    ForecastBatchWorkspace ws(c, 5);
    std::vector<float> input(5 * f::input.size()), output(5, 123);
    for (std::size_t b = 0; b < 5; ++b)
        std::copy(f::input.begin(), f::input.end(),
                  input.begin() + static_cast<std::ptrdiff_t>(b * f::input.size()));
    CHECK(model.forward_batch(input, 0, f::tokens, output, ws, SequenceOutput::last) ==
          Error::shape_mismatch);
    CHECK(model.forward_batch(input, std::numeric_limits<std::size_t>::max(), f::tokens, output, ws,
                              SequenceOutput::last) == Error::shape_mismatch);
    CHECK(model.forward_batch(input, 5, 0, output, ws, SequenceOutput::last) ==
          Error::shape_mismatch);
    CHECK(model.forward_batch(input, 5, f::tokens, output, ws, static_cast<SequenceOutput>(-1)) ==
          Error::shape_mismatch);
    CHECK(model.forward_batch(input, 5, f::tokens, std::span(output).first(4), ws,
                              SequenceOutput::last) == Error::shape_mismatch);
    ForecastBatchWorkspace small(c, 4);
    CHECK(model.forward_batch(input, 5, f::tokens, output, small, SequenceOutput::last) ==
          Error::shape_mismatch);
    auto other = c;
    other.output = ForecastOutput::correction;
    ForecastBatchWorkspace wrong(other, 5);
    CHECK(model.forward_batch(input, 5, f::tokens, output, wrong, SequenceOutput::last) ==
          Error::shape_mismatch);
    for (std::size_t i : {std::size_t{0}, input.size() / 2, input.size() - 1}) {
        const auto saved = input[i];
        input[i] = std::numeric_limits<float>::infinity();
        CHECK(model.forward_batch(input, 5, f::tokens, output, ws, SequenceOutput::last) ==
              Error::non_finite);
        input[i] = saved;
    }
    CHECK(model.forward_batch(input, 5, f::tokens, std::span(input).subspan(1, 5), ws,
                              SequenceOutput::last) == Error::aliasing);
    test::unchanged(output);
    ForecastBatchWorkspace moved(std::move(ws));
    CHECK(model.forward_batch(input, 5, f::tokens, output, ws, SequenceOutput::last) ==
          Error::invalid_state);
    CHECK(model.forward_batch(input, 5, f::tokens, output, moved, SequenceOutput::last) ==
          Error::ok);
    for (auto size : {std::size_t{0}, std::size_t{1025}, std::numeric_limits<std::size_t>::max()}) {
        bool caught = false;
        try {
            ForecastBatchWorkspace bad(c, size);
        } catch (const std::exception &) {
            caught = true;
        }
        CHECK(caught);
    }
    // A finite input in the LAST chunk overflows only during denormalization.
    // Even earlier successfully computed windows must not reach caller output.
    ForecastConfig late{1, 2, 1, 1, 2, 1, 0, 1e-5F, ForecastOutput::correction};
    std::vector<float> p(forecast_parameter_count(late), 0);
    p[1] = std::numeric_limits<float>::max();
    p.back() = 1;
    ForecastModel overflow(late, p);
    ForecastBatchWorkspace late_ws(late, 5);
    std::vector<float> values(10, 0), out(5, 123);
    values.back() = std::numeric_limits<float>::max();
    CHECK(overflow.forward_batch(values, 5, 2, out, late_ws, SequenceOutput::last) ==
          Error::numerical_error);
    test::unchanged(out);
    values.back() = 0;
    CHECK(overflow.forward_batch(values, 5, 2, out, late_ws, SequenceOutput::last) == Error::ok);
    for (auto v : out)
        CHECK(v == std::numeric_limits<float>::max());
}
void independent() {
    // Zero correction head gives the last observed target exactly, by construction.
    auto c = config();
    c.output = ForecastOutput::correction;
    auto p = f::parameters;
    std::fill(p.end() - static_cast<std::ptrdiff_t>(c.width + 1), p.end(), 0.0F);
    ForecastModel model(c, p);
    ForecastBatchWorkspace ws(c, 9);
    std::vector<float> input(9 * f::input.size()), out(9 * f::tokens);
    for (std::size_t i = 0; i < input.size(); ++i)
        input[i] = static_cast<float>(i % 31) / 8;
    CHECK(model.forward_batch(input, 9, f::tokens, out, ws) == Error::ok);
    for (std::size_t i = 0; i < out.size(); ++i)
        CHECK(out[i] == input[i * c.features + c.target]);
    // Other windows and later tokens must not affect an earlier window/prefix.
    ForecastModel full(config(), f::parameters);
    std::vector<float> before(out.size());
    CHECK(full.forward_batch(input, 9, f::tokens, before, ws) == Error::shape_mismatch);
    ForecastBatchWorkspace full_ws(config(), 9);
    CHECK(full.forward_batch(input, 9, f::tokens, before, full_ws) == Error::ok);
    for (std::size_t i = f::features * 2; i < input.size(); ++i)
        input[i] += 4;
    CHECK(full.forward_batch(input, 9, f::tokens, out, full_ws) == Error::ok);
    exact(std::span(out).first(2), std::span(before).first(2));
}
} // namespace
int main(int argc, char **argv) {
    try {
        CHECK(argc == 2);
        const std::string_view group(argv[1]);
        if (group == "parity")
            parity();
        else if (group == "block")
            block();
        else if (group == "shapes")
            shapes();
        else if (group == "failures")
            failures();
        else if (group == "independent")
            independent();
        else
            CHECK(false);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
