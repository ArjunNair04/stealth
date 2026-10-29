#include "stealth/attention.hpp"
#include "test_support.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

using namespace stealth::numeric;
namespace {
void hand() {
    AttentionWorkspace workspace(4, 2, 2);
    std::array<float, 8> zero{}, values{10, 2, 20, 4, 30, 8, 40, 10}, out{};
    CHECK(causal_attention(zero, zero, values, 4, out, workspace) == Error::ok);
    const std::array<float, 8> expected{10, 2, 15, 3, 20, 14.0F / 3, 25, 6};
    for (std::size_t i = 0; i < out.size(); ++i)
        test::near(out[i], expected[i], 2e-5, 5e-5);
    CHECK(workspace.bytes() == (4 * 2 + 3 * 4) * sizeof(float));
    // Unequal scores have a hand-derived 1:2 probability ratio at the second row.
    AttentionWorkspace one_head(2, 1, 1);
    std::array<float, 2> q{0, 1}, k{0, static_cast<float>(std::log(2.0))}, v{3, 9}, y{};
    CHECK(causal_attention(q, k, v, 2, y, one_head) == Error::ok);
    test::near(y[0], 3);
    test::near(y[1], 7);
    // A one-token call must select the current value, regardless of finite scores.
    CHECK(causal_attention(std::span(q).first(1), std::span(k).first(1), std::span(v).first(1), 1,
                           std::span(y).first(1), one_head) == Error::ok);
    test::near(y[0], 3);
}
void failures() {
    for (const auto dims : {std::array<std::size_t, 3>{0, 4, 2}, {5, 0, 1}, {5, 4, 0}, {5, 4, 3}})
        test::throws<std::invalid_argument>(
            [&] { AttentionWorkspace bad(dims[0], dims[1], dims[2]); });
    test::throws<std::length_error>([] { AttentionWorkspace bad(max_context + 1, 4, 2); });
    test::throws<std::length_error>([] { AttentionWorkspace bad(2, max_width + 1, 1); });
    test::throws<std::length_error>(
        [] { AttentionWorkspace bad(std::numeric_limits<std::size_t>::max(), 4, 2); });
    AttentionWorkspace ws(2, 2, 1);
    std::array<float, 4> q{1, 2, 3, 4}, k{2, 3, 4, 5}, v{3, 4, 5, 6}, out{123, 123, 123, 123};
    CHECK(causal_attention(q, k, v, 0, out, ws) == Error::shape_mismatch);
    test::unchanged(out);
    CHECK(causal_attention(q, k, v, 3, out, ws) == Error::shape_mismatch);
    test::unchanged(out);
    CHECK(causal_attention(std::span(q).first(3), k, v, 2, out, ws) == Error::shape_mismatch);
    test::unchanged(out);
    for (auto invalid :
         {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
        for (int field = 0; field < 3; ++field) {
            auto a = q, b = k, c = v;
            if (field == 0)
                a[3] = invalid;
            if (field == 1)
                b[3] = invalid;
            if (field == 2)
                c[3] = invalid;
            CHECK(causal_attention(a, b, c, 2, out, ws) == Error::non_finite);
            test::unchanged(out);
        }
    }
    const auto before = q;
    CHECK(causal_attention(q, k, v, 2, q, ws) == Error::aliasing);
    CHECK(q == before);
    std::array<float, 6> overlap{1, 2, 3, 4, 5, 6};
    const auto old = overlap;
    CHECK(causal_attention(std::span(overlap).first(4), k, v, 2, std::span(overlap).subspan(2),
                           ws) == Error::aliasing);
    CHECK(overlap == old);
    auto huge = q;
    huge[3] = std::numeric_limits<float>::max();
    CHECK(causal_attention(huge, huge, v, 2, out, ws) == Error::numerical_error);
    test::unchanged(out); // The overflow occurs after earlier rows have been computed.
    CHECK(causal_attention(q, k, v, 2, out, ws) == Error::ok);
    auto moved = std::move(ws);
    out.fill(123);
    CHECK(causal_attention(q, k, v, 2, out, ws) == Error::invalid_state);
    test::unchanged(out);
    CHECK(causal_attention(q, k, v, 2, out, moved) == Error::ok);
}
void causality() {
    constexpr std::size_t n = 7, width = 4;
    AttentionWorkspace ws(n, width, 2), other(n, width, 2);
    std::array<float, n * width> q{}, k{}, v{}, original{}, changed{}, prefix{};
    test::Random random;
    for (auto *a : {&q, &k, &v})
        for (auto &x : *a)
            x = random.next();
    CHECK(causal_attention(q, k, v, n, original, ws) == Error::ok);
    for (std::size_t count = 1; count <= n; ++count) {
        CHECK(causal_attention(std::span(q).first(count * width), std::span(k).first(count * width),
                               std::span(v).first(count * width), count,
                               std::span(prefix).first(count * width), other) == Error::ok);
        for (std::size_t i = 0; i < count * width; ++i)
            test::near(prefix[i], original[i], 0, 0);
        auto a = q, b = k, c = v;
        for (std::size_t i = count * width; i < q.size(); ++i) {
            a[i] += 10;
            b[i] -= 20;
            c[i] += 30;
        }
        CHECK(causal_attention(a, b, c, n, changed, ws) == Error::ok);
        for (std::size_t i = 0; i < count * width; ++i)
            test::near(changed[i], original[i], 0, 0);
    }
    AttentionWorkspace boundary(max_context, 1, 1);
    std::vector<float> zeros(max_context), ones(max_context, 1), result(max_context);
    CHECK(causal_attention(zeros, zeros, ones, max_context, result, boundary) == Error::ok);
    for (float x : result)
        test::near(x, 1, 2e-5, 5e-5);
}
void generated() {
    test::Random random;
    for (std::size_t width : {1U, 3U, 4U, 8U, 17U})
        for (std::size_t heads : {1U, 2U, 3U, 4U, 17U}) {
            if (width % heads != 0)
                continue;
            for (std::size_t n : {1U, 2U, 5U, 9U}) {
                AttentionWorkspace ws(n, width, heads);
                std::vector<float> q(n * width), k(n * width), v(n * width), out(n * width);
                for (auto *a : {&q, &k, &v})
                    for (auto &x : *a)
                        x = random.next();
                CHECK(causal_attention(q, k, v, n, out, ws) == Error::ok);
                const auto d = width / heads;
                // Independent unshifted double softmax is safe for this bounded fixture.
                for (std::size_t t = 0; t < n; ++t)
                    for (std::size_t h = 0; h < heads; ++h) {
                        std::vector<double> weights(t + 1);
                        double total = 0;
                        for (std::size_t j = 0; j <= t; ++j) {
                            double dot = 0;
                            for (std::size_t c = 0; c < d; ++c)
                                dot += static_cast<double>(q[t * width + h * d + c]) *
                                       k[j * width + h * d + c];
                            weights[j] = std::exp(dot / std::sqrt(static_cast<double>(d)));
                            total += weights[j];
                        }
                        for (std::size_t c = 0; c < d; ++c) {
                            double answer = 0;
                            for (std::size_t j = 0; j <= t; ++j)
                                answer += weights[j] / total * v[j * width + h * d + c];
                            test::near(out[t * width + h * d + c], answer, 2e-5, 5e-5);
                        }
                    }
            }
        }
}
void tiled() {
    test::throws<std::invalid_argument>(
        [] { AttentionWorkspace bad(1, 8, 1, static_cast<AttentionKernel>(99)); });
    test::Random random;
    const auto compare = [&](std::size_t tokens, std::size_t width, std::size_t heads) {
        AttentionWorkspace reference(tokens, width, heads, AttentionKernel::reference);
        AttentionWorkspace optimized(tokens, width, heads, AttentionKernel::tiled);
        AttentionWorkspace automatic(tokens, width, heads);
        CHECK(reference.bytes() == optimized.bytes());
        std::vector<float> q(tokens * width), k(tokens * width), v(tokens * width);
        std::vector<float> expected(tokens * width), actual(tokens * width),
            default_output(tokens * width);
        for (auto *values : {&q, &k, &v})
            for (auto &value : *values)
                value = random.next() * 3.0F;
        CHECK(causal_attention(q, k, v, tokens, expected, reference) == Error::ok);
        CHECK(causal_attention(q, k, v, tokens, actual, optimized) == Error::ok);
        CHECK(causal_attention(q, k, v, tokens, default_output, automatic) == Error::ok);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            CHECK(std::bit_cast<std::uint32_t>(actual[i]) ==
                  std::bit_cast<std::uint32_t>(expected[i]));
            CHECK(std::bit_cast<std::uint32_t>(default_output[i]) ==
                  std::bit_cast<std::uint32_t>(expected[i]));
        }
        // Future changes cannot alter earlier rows, including tile/tail boundaries.
        if (tokens > 1) {
            auto changed = v;
            for (std::size_t i = width; i < changed.size(); ++i)
                changed[i] += 100;
            CHECK(causal_attention(q, k, changed, tokens, actual, optimized) == Error::ok);
            for (std::size_t i = 0; i < width; ++i)
                CHECK(actual[i] == expected[i]);
        }
        // Distinctive head/channel values independently expose cross-head loads.
        std::fill(q.begin(), q.end(), 0.0F);
        std::fill(k.begin(), k.end(), 0.0F);
        for (std::size_t t = 0; t < tokens; ++t)
            for (std::size_t c = 0; c < width; ++c)
                v[t * width + c] = static_cast<float>(c + 1);
        CHECK(causal_attention(q, k, v, tokens, actual, optimized) == Error::ok);
        for (std::size_t i = 0; i < actual.size(); ++i)
            test::near(actual[i], static_cast<double>(i % width + 1), 2e-5, 5e-5);
        for (auto kernel : {AttentionKernel::reference, AttentionKernel::tiled}) {
            AttentionWorkspace ws(tokens, width, heads, kernel);
            std::fill(actual.begin(), actual.end(), 123.0F);
            auto bad = q;
            bad.back() = std::numeric_limits<float>::infinity();
            CHECK(causal_attention(bad, k, v, tokens, actual, ws) == Error::non_finite);
            test::unchanged(actual);
            std::fill(q.begin(), q.end(), std::numeric_limits<float>::max());
            std::fill(k.begin(), k.end(), std::numeric_limits<float>::max());
            CHECK(causal_attention(q, k, v, tokens, actual, ws) == Error::numerical_error);
            test::unchanged(actual);
            std::fill(q.begin(), q.end(), 0.0F);
            std::fill(k.begin(), k.end(), 0.0F);
        }
    };
    for (std::size_t head_width : {1U, 3U, 7U, 8U, 9U, 15U, 16U, 17U, 31U, 32U, 63U, 64U, 65U, 127U,
                                   128U, 255U, 256U, 511U, 512U})
        for (std::size_t heads : {1U, 3U, 4U})
            if (head_width * heads <= max_width)
                for (std::size_t tokens : {1U, 5U, 17U})
                    compare(tokens, head_width * heads, heads);
    compare(96, 64, 4);
    compare(max_context, 8, 1);
}
} // namespace
int main(int argc, char *argv[]) {
    try {
        if (argc != 2)
            throw std::runtime_error("Expected test group");
        const std::string_view group(argv[1]);
        if (group == "hand")
            hand();
        else if (group == "failures")
            failures();
        else if (group == "causality")
            causality();
        else if (group == "generated")
            generated();
        else if (group == "tiled")
            tiled();
        else
            throw std::runtime_error("Unknown test group");
        std::cout << group << ": " << test::checks << " checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
