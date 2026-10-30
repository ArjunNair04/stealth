#include "fixtures/feed_forward.hpp"
#include "stealth/feed_forward.hpp"
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
void identical(std::span<const float> a, std::span<const float> b) {
    CHECK(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i)
        CHECK(std::bit_cast<std::uint32_t>(a[i]) == std::bit_cast<std::uint32_t>(b[i]));
}
void dense() {
    // Independent hand answer for two different rows.
    Dense hand(3, 2, std::array<float, 6>{1, 2, 3, -1, 0, 2}, std::array<float, 2>{0.5F, -0.5F});
    std::array<float, 6> x{2, -1, 0.5F, 1, 2, -1};
    std::array<float, 4> y{}, scratch{};
    CHECK(hand.forward_rows(x, 2, y, scratch) == Error::ok);
    identical(y, std::array<float, 4>{2, -1.5F, 2.5F, -3.5F});
    test::Random rng;
    for (std::size_t inputs : {1U, 3U, 8U, 17U, 64U, 256U})
        for (std::size_t outputs : {1U, 7U, 8U, 9U, 16U, 17U, 64U, 192U, 256U}) {
            std::vector<float> w(inputs * outputs), bias(outputs);
            for (auto &v : w)
                v = rng.next() / 8;
            for (auto &v : bias)
                v = rng.next() / 8;
            Dense scalar(inputs, outputs, w, bias, DenseKernel::reference);
            Dense packed(inputs, outputs, w, bias);
            for (std::size_t rows : {1U, 2U, 3U, 4U, 5U, 7U, 8U, 9U, 17U}) {
                std::vector<float> in(rows * inputs), expected(rows * outputs),
                    actual(rows * outputs), reference(rows * outputs), temp(rows * outputs);
                for (auto &v : in)
                    v = rng.next();
                for (std::size_t row = 0; row < rows; ++row)
                    CHECK(scalar.forward(std::span(in).subspan(row * inputs, inputs),
                                         std::span(expected).subspan(row * outputs, outputs),
                                         temp) == Error::ok);
                CHECK(scalar.forward_rows(in, rows, reference, temp) == Error::ok);
                CHECK(packed.forward_rows(in, rows, actual, temp) == Error::ok);
                identical(expected, reference);
                identical(expected, actual);
                // Editing one observation cannot alter any other row.
                in.back() += 1.0F;
                CHECK(packed.forward_rows(in, rows, actual, temp) == Error::ok);
                identical(std::span(expected).first((rows - 1) * outputs),
                          std::span(actual).first((rows - 1) * outputs));
            }
        }
    // Non-binary fractions and cancellation expose reordered accumulation that
    // the small dyadic fixtures can miss. Cover both row and column tile edges.
    for (std::size_t outputs : {7U, 8U, 9U, 15U, 16U, 17U}) {
        constexpr std::size_t inputs = 37;
        std::vector<float> w(inputs * outputs), bias(outputs);
        for (auto &v : w)
            v = rng.next() / 7.1F;
        for (auto &v : bias)
            v = rng.next() / 3.7F;
        Dense scalar(inputs, outputs, w, bias, DenseKernel::reference);
        Dense packed(inputs, outputs, w, bias);
        for (std::size_t rows : {1U, 3U, 4U, 5U, 8U, 9U}) {
            std::vector<float> in(rows * inputs), expected(rows * outputs),
                actual(rows * outputs), temp(rows * outputs);
            for (std::size_t i = 0; i < in.size(); ++i)
                in[i] = rng.next() / 1.3F * (i % 3 == 0 ? 10000.0F : 0.001F);
            CHECK(scalar.forward_rows(in, rows, expected, temp) == Error::ok);
            CHECK(packed.forward_rows(in, rows, actual, temp) == Error::ok);
            identical(expected, actual);
        }
    }
}
void failures() {
    for (auto kernel : {DenseKernel::reference, DenseKernel::packed}) {
        Dense d(1, 1, std::array<float, 1>{2}, std::array<float, 1>{0}, kernel);
        std::array<float, 5> x{1, 2, 3, 4, 5}, y{}, scratch{};
        y.fill(123.0F);
        CHECK(d.forward_rows(x, 0, y, scratch) == Error::shape_mismatch);
        CHECK(d.forward_rows(x, std::numeric_limits<std::size_t>::max(), y, scratch) ==
              Error::shape_mismatch);
        CHECK(d.forward_rows(x, 4, y, scratch) == Error::shape_mismatch);
        CHECK(d.forward_rows(x, 5, y, std::span(scratch).first(4)) == Error::insufficient_scratch);
        CHECK(d.forward_rows(x, 5, y, y) == Error::aliasing);
        CHECK(d.forward_rows(x, 5, y, x) == Error::aliasing);
        test::unchanged(y);
        const auto old = x;
        CHECK(d.forward_rows(x, 5, x, scratch) == Error::aliasing);
        CHECK(x == old);
        for (float bad :
             {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
            x.back() = bad;
            CHECK(d.forward_rows(x, 5, y, scratch) == Error::non_finite);
            test::unchanged(y);
        }
        x.back() = std::numeric_limits<float>::max();
        CHECK(d.forward_rows(x, 5, y, scratch) == Error::numerical_error);
        test::unchanged(y); // Failure in last row must preserve earlier rows too.
        x.back() = 5;
        CHECK(d.forward_rows(x, 5, y, scratch) == Error::ok);
        auto moved = std::move(d);
        CHECK(d.forward_rows(x, 5, y, scratch) == Error::invalid_state);
        CHECK(moved.forward_rows(x, 5, y, scratch) == Error::ok);
        std::array<float, 15> overlap{};
        // Scratch's unused tail still aliases input and must be rejected.
        CHECK(moved.forward_rows(std::span(overlap).subspan(10, 5), 5, y,
                                 std::span(overlap).subspan(4)) == Error::aliasing);
    }
}
void tile_failures() {
    for (std::size_t outputs : {7U, 8U, 9U, 15U, 16U, 17U}) {
        std::vector<float> weights(outputs * 2, 0.0F), bias(outputs, 0.0F);
        weights[(outputs - 1) * 2] = 2.0F;
        Dense model(2, outputs, weights, bias);
        for (std::size_t rows : {1U, 3U, 4U, 5U, 8U, 9U}) {
            std::vector<float> in(rows * 2, 1.0F), out(rows * outputs, 123.0F),
                scratch(rows * outputs);
            // A failure at the last output of the last row cannot publish any
            // earlier successful tile, including a partial row/column tile.
            in[(rows - 1) * 2] = std::numeric_limits<float>::max();
            CHECK(model.forward_rows(in, rows, out, scratch) == Error::numerical_error);
            test::unchanged(out);
        }
    }
}
void feed_forward() {
    FeedForwardParameters p{fixture::scale,   fixture::shift,        fixture::up_weights,
                            fixture::up_bias, fixture::down_weights, fixture::down_bias};
    for (auto kernel : {DenseKernel::reference, DenseKernel::packed}) {
        FeedForward model(fixture::width, fixture::hidden, p, fixture::epsilon, kernel);
        FeedForwardWorkspace row_ws(fixture::width, fixture::hidden);
        for (std::size_t rows : {1U, 3U, 4U, 5U, 8U, 9U}) {
            FeedForwardWorkspace ws(fixture::width, fixture::hidden, rows);
            CHECK(ws.bytes() == row_ws.bytes() * rows);
            std::vector<float> in(rows * fixture::width), expected(in.size()),
                out(in.size(), 123.0F);
            for (std::size_t row = 0; row < rows; ++row) {
                const auto &values = row % 2 == 0 ? fixture::input : fixture::input_other;
                const auto &answer = row % 2 == 0 ? fixture::expected : fixture::expected_other;
                std::copy(values.begin(), values.end(),
                          in.begin() + static_cast<std::ptrdiff_t>(row * fixture::width));
                auto dest = std::span(expected).subspan(row * fixture::width, fixture::width);
                CHECK(model.forward(values, dest, row_ws) == Error::ok);
                for (std::size_t i = 0; i < dest.size(); ++i)
                    test::near(dest[i], answer[i], 2e-5, 5e-5);
            }
            CHECK(model.forward_rows(in, rows, out, ws) == Error::ok);
            identical(expected, out);
            std::fill(out.begin(), out.end(), 123.0F);
            CHECK(model.forward_rows(in, rows + 1, out, ws) == Error::shape_mismatch);
            CHECK(model.forward_rows(in, 0, out, ws) == Error::shape_mismatch);
            CHECK(model.forward_rows(in, rows, in, ws) == Error::aliasing);
            in.back() = std::numeric_limits<float>::infinity();
            CHECK(model.forward_rows(in, rows, out, ws) == Error::non_finite);
            test::unchanged(out);
            auto moved = std::move(ws);
            CHECK(model.forward_rows(in, rows, out, ws) == Error::invalid_state);
            in.back() = 0;
            CHECK(model.forward_rows(in, rows, out, moved) == Error::ok);
        }
    }
    test::throws<std::invalid_argument>([] { FeedForwardWorkspace ws(4, 7, 0); });
    test::throws<std::length_error>(
        [] { FeedForwardWorkspace ws(4, 7, std::numeric_limits<std::size_t>::max()); });
    const std::array<float, 1> zero{0}, one{1}, huge{std::numeric_limits<float>::max() * 0.5F};
    FeedForward late(1, 1, {zero, zero, zero, one, huge, zero});
    FeedForwardWorkspace ws(1, 1, 5);
    std::array<float, 5> in{}, out{};
    out.fill(123.0F);
    in.back() = std::numeric_limits<float>::max();
    CHECK(late.forward_rows(in, 5, out, ws) == Error::numerical_error);
    test::unchanged(out);
}
} // namespace
int main(int argc, char *argv[]) {
    try {
        if (argc != 2)
            throw std::runtime_error("Expected group");
        const std::string_view group(argv[1]);
        if (group == "dense")
            dense();
        else if (group == "failures") {
            failures();
            tile_failures();
        }
        else if (group == "feed_forward")
            feed_forward();
        else
            throw std::runtime_error("Unknown group");
        std::cout << group << ": " << test::checks << " checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
