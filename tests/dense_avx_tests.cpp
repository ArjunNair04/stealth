#include "../src/dense_avx.hpp"
#include "stealth/numeric.hpp"
#include "test_support.hpp"
#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <vector>
using namespace stealth::numeric;
int main() {
    try {
        test::Random rng;
        for (std::size_t inputs : {1U, 3U, 7U, 16U, 63U, 64U, 65U, 256U})
            for (std::size_t outputs : {8U, 9U, 15U, 16U, 17U, 63U, 64U, 192U}) {
                const auto padded = ((outputs + 7) / 8) * 8;
                std::vector<float> weights(inputs * outputs), bias(outputs),
                    packed(inputs * padded);
                for (auto &v : weights)
                    v = rng.next() / 13.1F;
                for (auto &v : bias)
                    v = rng.next() / 7.3F;
                for (std::size_t o = 0; o < outputs; ++o)
                    for (std::size_t i = 0; i < inputs; ++i)
                        packed[(o / 8) * inputs * 8 + i * 8 + o % 8] = weights[o * inputs + i];
                Dense reference(inputs, outputs, weights, bias, DenseKernel::reference);
                Dense automatic(inputs, outputs, weights, bias);
                for (std::size_t rows : {1U, 2U, 3U, 4U, 5U, 7U, 8U, 9U}) {
                    // Offset all caller buffers by one float: no alignment requirement.
                    std::vector<float> x(1 + rows * inputs), y(2 + rows * outputs, 123.0F),
                        expected(rows * outputs), scratch(rows * outputs), actual(rows * outputs);
                    for (auto &v : x)
                        v = rng.next() / 3.7F;
                    auto in = std::span(x).subspan(1);
                    auto out = std::span(y).subspan(1, rows * outputs);
                    CHECK(reference.forward_rows(in, rows, expected, scratch) == Error::ok);
                    const bool used = stealth::numeric::numeric_detail::try_dense_avx(
                        in.data(), packed.data(), bias.data(), out.data(), inputs, outputs, rows);
                    CHECK(used == stealth::numeric::numeric_detail::dense_avx_available());
                    for (std::size_t i = 0; i < out.size(); ++i) {
                        if (used)
                            CHECK(std::bit_cast<std::uint32_t>(out[i]) ==
                                  std::bit_cast<std::uint32_t>(expected[i]));
                        else
                            CHECK(out[i] == 123.0F);
                    }
                    CHECK(y.front() == 123.0F && y.back() == 123.0F);
                    CHECK(automatic.forward_rows(in, rows, actual, scratch) == Error::ok);
                    for (std::size_t i = 0; i < actual.size(); ++i)
                        CHECK(std::bit_cast<std::uint32_t>(actual[i]) ==
                              std::bit_cast<std::uint32_t>(expected[i]));
                }
            }
        // A fused multiply/add would produce -2^-46 rather than exact zero.
        // This hand case guards the arithmetic contract without a second kernel.
        for (std::size_t rows : {1U, 4U, 5U}) {
            constexpr std::size_t inputs = 64, outputs = 17;
            std::vector<float> w(inputs * outputs, 0.0F), bias(outputs, -1.0F),
                x(rows * inputs, 0.0F), y(rows * outputs, 123.0F), scratch(y.size());
            for (std::size_t o = 0; o < outputs; ++o)
                w[o * inputs] = 0x1.000002p0F;
            for (std::size_t r = 0; r < rows; ++r)
                x[r * inputs] = 0x1.fffffcp-1F;
            Dense model(inputs, outputs, w, bias);
            CHECK(model.forward_rows(x, rows, y, scratch) == Error::ok);
            for (float value : y)
                CHECK(std::bit_cast<std::uint32_t>(value) == 0);
        }
        // Exact signed zero, subnormal and range endpoints use no approximation.
        for (float value :
             {0.0F, -0.0F, std::numeric_limits<float>::denorm_min(),
              -std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::max(),
              -std::numeric_limits<float>::max()}) {
            std::vector<float> weights(17, 1.0F), bias(17, std::signbit(value) ? -0.0F : 0.0F),
                input(5, value), output(85), scratch(85);
            Dense model(1, 17, weights, bias);
            CHECK(model.forward_rows(input, 5, output, scratch) == Error::ok);
            for (float actual : output)
                CHECK(std::bit_cast<std::uint32_t>(actual) == std::bit_cast<std::uint32_t>(value));
        }
        std::cout << "AVX available: " << stealth::numeric::numeric_detail::dense_avx_available()
                  << "; " << test::checks << " checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
