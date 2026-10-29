#include "fixtures/attention.hpp"
#include "stealth/transformer_block.hpp"
#include "test_support.hpp"
#include <algorithm>
#include <array>
#include <limits>
#include <string_view>
#include <vector>

using namespace stealth::numeric;
namespace f = attention_fixture;
namespace {
TransformerParameters parameters() {
    return {f::attn_scale,
            f::attn_shift,
            f::qkv_weights,
            f::qkv_bias,
            f::projection_weights,
            f::projection_bias,
            {f::ff_scale, f::ff_shift, f::up_weights, f::up_bias, f::down_weights, f::down_bias}};
}
void compare(std::span<const float> actual, std::span<const float> expected) {
    CHECK(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
        test::near(actual[i], expected[i], 3e-5, 1e-4);
}
void parity() {
    AttentionWorkspace attention(f::tokens, f::width, f::heads);
    std::array<float, f::tokens * f::width> out{};
    CHECK(causal_attention(f::query, f::key, f::value, f::tokens, out, attention) == Error::ok);
    for (std::size_t i = 0; i < out.size(); ++i)
        test::near(out[i], f::context[i], 2e-5, 5e-5);
    for (auto kernel : {DenseKernel::reference, DenseKernel::packed}) {
        TransformerBlock block(f::width, f::heads, f::hidden, parameters(), f::epsilon, kernel);
        TransformerWorkspace ws(f::tokens, f::width, f::heads, f::hidden);
        CHECK(block.forward(f::input, f::tokens, out, ws) == Error::ok);
        compare(out, f::expected);
        CHECK(block.forward(f::input_other, f::tokens, out, ws) == Error::ok);
        compare(out, f::expected_other);
        CHECK(block.parameter_bytes() > 0);
        CHECK(ws.bytes() > 0);
    }
}
void causality() {
    TransformerBlock block(f::width, f::heads, f::hidden, parameters());
    TransformerWorkspace ws(f::tokens, f::width, f::heads, f::hidden),
        other(f::tokens, f::width, f::heads, f::hidden);
    std::array<float, f::tokens * f::width> original{}, out{}, prefix{};
    CHECK(block.forward(f::input, f::tokens, original, ws) == Error::ok);
    for (std::size_t n = 1; n <= f::tokens; ++n) {
        CHECK(block.forward(std::span(f::input).first(n * f::width), n,
                            std::span(prefix).first(n * f::width), other) == Error::ok);
        for (std::size_t i = 0; i < n * f::width; ++i)
            test::near(prefix[i], original[i], 0, 0);
        auto input = f::input;
        for (std::size_t i = n * f::width; i < input.size(); ++i)
            input[i] = static_cast<float>(i) * 7;
        CHECK(block.forward(input, f::tokens, out, ws) == Error::ok);
        for (std::size_t i = 0; i < n * f::width; ++i)
            test::near(out[i], original[i], 0, 0);
    }
}
void failures() {
    TransformerBlock block(f::width, f::heads, f::hidden, parameters());
    TransformerWorkspace ws(f::tokens, f::width, f::heads, f::hidden),
        wrong_heads(f::tokens, f::width, 1, f::hidden),
        wrong_hidden(f::tokens, f::width, f::heads, 8);
    std::array<float, f::tokens * f::width> out{};
    out.fill(123);
    CHECK(block.forward(f::input, f::tokens, out, wrong_heads) == Error::shape_mismatch);
    test::unchanged(out);
    CHECK(block.forward(f::input, f::tokens, out, wrong_hidden) == Error::shape_mismatch);
    test::unchanged(out);
    for (std::size_t n :
         {std::size_t{0}, std::size_t{5}, std::numeric_limits<std::size_t>::max()}) {
        CHECK(block.forward(f::input, n, out, ws) == Error::shape_mismatch);
        test::unchanged(out);
    }
    CHECK(block.forward(std::span(f::input).first(3), f::tokens, out, ws) == Error::shape_mismatch);
    test::unchanged(out);
    auto input = f::input;
    input.back() = std::numeric_limits<float>::quiet_NaN();
    CHECK(block.forward(input, f::tokens, out, ws) == Error::non_finite);
    test::unchanged(out);
    input = f::input;
    CHECK(block.forward(input, f::tokens, input, ws) == Error::aliasing);
    CHECK(input == f::input);
    std::array<float, f::tokens * f::width + 2> overlap{};
    CHECK(block.forward(std::span(overlap).first(f::input.size()), f::tokens,
                        std::span(overlap).subspan(2), ws) == Error::aliasing);
    for (float x : overlap)
        CHECK(x == 0);
    test::throws<std::invalid_argument>([&] { TransformerBlock bad(4, 3, 7, parameters()); });
    test::throws<std::invalid_argument>([&] { TransformerBlock bad(4, 2, 0, parameters()); });
    test::throws<std::invalid_argument>([&] { TransformerBlock bad(4, 2, 7, parameters(), 0); });
    test::throws<std::length_error>(
        [&] { TransformerBlock bad(max_width + 1, 1, 7, parameters()); });
    test::throws<std::length_error>([&] { TransformerWorkspace bad(4, 4, 2, max_hidden + 1); });
    auto p = parameters();
    p.attention_scale = std::span(f::attn_scale).first(3);
    test::throws<std::invalid_argument>([&] { TransformerBlock bad(4, 2, 7, p); });
    auto scale = f::attn_scale;
    scale[0] = std::numeric_limits<float>::infinity();
    p = parameters();
    p.attention_scale = scale;
    test::throws<std::invalid_argument>([&] { TransformerBlock bad(4, 2, 7, p); });
    // The attention projection overflows after normalization and attention have run.
    auto weights = f::projection_weights;
    weights.fill(std::numeric_limits<float>::max());
    auto bias = f::qkv_bias;
    std::fill(bias.begin() + 8, bias.end(), 2.0F);
    auto qkv = f::qkv_weights;
    std::fill(qkv.begin(), qkv.end(), 0.0F);
    p = parameters();
    p.projection_weights = weights;
    p.qkv_weights = qkv;
    p.qkv_bias = bias;
    TransformerBlock overflow(4, 2, 7, p);
    CHECK(overflow.forward(f::input, f::tokens, out, ws) == Error::numerical_error);
    test::unchanged(out);
    // First rows succeed internally; a residual overflows only on the final row.
    auto zeros = f::projection_weights;
    zeros.fill(0);
    auto huge_bias = f::projection_bias;
    huge_bias.fill(std::numeric_limits<float>::max());
    p = parameters();
    p.projection_weights = zeros;
    p.projection_bias = huge_bias;
    TransformerBlock late(4, 2, 7, p);
    auto late_input = f::input;
    std::fill(late_input.end() - 4, late_input.end(), std::numeric_limits<float>::max());
    CHECK(late.forward(late_input, f::tokens, out, ws) == Error::numerical_error);
    test::unchanged(out);
    CHECK(block.forward(f::input, f::tokens, out, ws) == Error::ok);
    compare(out, f::expected);
    auto moved_ws = std::move(ws);
    out.fill(123);
    CHECK(block.forward(f::input, f::tokens, out, ws) == Error::invalid_state);
    test::unchanged(out);
    auto moved = std::move(block);
    CHECK(block.forward(f::input, f::tokens, out, moved_ws) == Error::invalid_state);
    test::unchanged(out);
    CHECK(moved.forward(f::input, f::tokens, out, moved_ws) == Error::ok);
    compare(out, f::expected);
}
void replay() {
    TransformerBlock block(f::width, f::heads, f::hidden, parameters());
    TransformerWorkspace a(f::tokens, f::width, f::heads, f::hidden),
        b(f::tokens, f::width, f::heads, f::hidden);
    std::array<float, f::tokens * f::width> x{}, y{};
    for (int i = 0; i < 2000; ++i) {
        CHECK(block.forward(f::input, f::tokens, x, a) == Error::ok);
        compare(x, f::expected);
        CHECK(block.forward(f::input_other, f::tokens, y, b) == Error::ok);
        compare(y, f::expected_other);
    }
}
void generated() {
    test::Random random;
    for (std::size_t width : {1U, 3U, 8U, 64U}) {
        const std::size_t hidden = width * 4, heads = width == 64 ? 4 : 1, tokens = 5;
        std::vector<float> scale(width, 1), shift(width), qkv(3 * width * width), qb(3 * width),
            pw(width * width), pb(width), up(width * hidden), ub(hidden), down(hidden * width),
            db(width), input(tokens * width), a(tokens * width), b(tokens * width);
        for (auto *v : {&qkv, &qb, &pw, &pb, &up, &ub, &down, &db, &input})
            for (float &x : *v)
                x = random.next() / 8;
        TransformerParameters p{scale, shift, qkv, qb, pw, pb, {scale, shift, up, ub, down, db}};
        TransformerBlock scalar(width, heads, hidden, p, 1e-5F, DenseKernel::reference),
            packed(width, heads, hidden, p);
        TransformerWorkspace wa(tokens, width, heads, hidden), wb(tokens, width, heads, hidden);
        CHECK(scalar.forward(input, tokens, a, wa) == Error::ok);
        CHECK(packed.forward(input, tokens, b, wb) == Error::ok);
        compare(a, b);
    }
}
} // namespace
int main(int argc, char *argv[]) {
    try {
        if (argc != 2)
            throw std::runtime_error("Expected test group");
        const std::string_view g(argv[1]);
        if (g == "parity")
            parity();
        else if (g == "causality")
            causality();
        else if (g == "failures")
            failures();
        else if (g == "replay")
            replay();
        else if (g == "generated")
            generated();
        else
            throw std::runtime_error("Unknown group");
        std::cout << g << ": " << test::checks << " checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
