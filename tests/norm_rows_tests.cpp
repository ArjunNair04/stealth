#include "stealth/numeric.hpp"
#include "test_support.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <vector>
using namespace stealth::numeric;
int main() {
    try {
        const std::array<float, 4> hand{1, 3, 2, 6};
        const std::array<float, 2> scale{2, 3}, shift{4, 5};
        std::array<float, 4> out{}, scratch{};
        CHECK(layer_norm_rows(hand, 2, scale, shift, 1e-5F, out, scratch) == Error::ok);
        for (std::size_t r = 0; r < 2; ++r) {
            const double delta = static_cast<double>(r + 1);
            test::near(out[r * 2], 4 - 2 * delta / std::sqrt(delta * delta + 1e-5F), 2e-6, 2e-5);
            test::near(out[r * 2 + 1], 5 + 3 * delta / std::sqrt(delta * delta + 1e-5F), 2e-6,
                       2e-5);
        }
        test::Random rng;
        for (std::size_t width : {1U, 2U, 7U, 8U, 33U, 64U, 513U})
            for (std::size_t rows : {1U, 2U, 3U, 4U, 5U, 7U, 8U, 9U, 32U, 33U}) {
                std::vector<float> x(rows * width), a(width), b(width), y(x.size()),
                    expected(x.size()), temp(x.size());
                for (auto &v : x)
                    v = rng.next() / 3.7F;
                for (auto &v : a)
                    v = rng.next() / 1.3F;
                for (auto &v : b)
                    v = rng.next() / 11.1F;
                for (std::size_t r = 0; r < rows; ++r)
                    CHECK(layer_norm(std::span(x).subspan(r * width, width), a, b, 1e-5F,
                                     std::span(expected).subspan(r * width, width),
                                     temp) == Error::ok);
                CHECK(layer_norm_rows(x, rows, a, b, 1e-5F, y, temp) == Error::ok);
                for (std::size_t i = 0; i < y.size(); ++i)
                    CHECK(std::bit_cast<std::uint32_t>(y[i]) ==
                          std::bit_cast<std::uint32_t>(expected[i]));
            }
        std::array<float, 10> x{}, y{}, temp{};
        std::array<float, 2> a{1, 1}, b{0, 0};
        y.fill(123);
        CHECK(layer_norm_rows(x, 0, a, b, 1e-5F, y, temp) == Error::shape_mismatch);
        CHECK(layer_norm_rows(x, std::numeric_limits<std::size_t>::max(), a, b, 1e-5F, y, temp) ==
              Error::shape_mismatch);
        CHECK(layer_norm_rows(x, 5, a, b, 0, y, temp) == Error::shape_mismatch);
        CHECK(layer_norm_rows(x, 5, a, b, 1e-5F, y, std::span(temp).first(9)) ==
              Error::insufficient_scratch);
        CHECK(layer_norm_rows(x, 5, a, b, 1e-5F, std::span(x), temp) == Error::aliasing);
        CHECK(layer_norm_rows(x, 5, a, b, 1e-5F, y, std::span(y)) == Error::aliasing);
        for (std::size_t i = 0; i < x.size(); ++i) {
            x[i] = std::numeric_limits<float>::quiet_NaN();
            CHECK(layer_norm_rows(x, 5, a, b, 1e-5F, y, temp) == Error::non_finite);
            x[i] = 0;
        }
        // All first four rows are valid; the last row overflows after scaling/shift.
        a[0] = b[0] = std::numeric_limits<float>::max();
        x[8] = 1;
        x[9] = -1;
        CHECK(layer_norm_rows(x, 5, a, b, 1e-5F, y, temp) == Error::numerical_error);
        test::unchanged(y);
        x[8] = x[9] = 0;
        CHECK(layer_norm_rows(x, 5, a, b, 1e-5F, y, temp) == Error::ok);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
