#include "stealth/numeric.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string_view>
#include <type_traits>
#include <vector>

using namespace stealth::numeric;
namespace {
constexpr float nan = std::numeric_limits<float>::quiet_NaN();
constexpr float inf = std::numeric_limits<float>::infinity();
constexpr float largest = std::numeric_limits<float>::max();

void buffers() {
    static_assert(!std::is_copy_constructible_v<AlignedBuffer>);
    static_assert(std::is_nothrow_move_constructible_v<AlignedBuffer>);
    AlignedBuffer empty;
    CHECK(empty.size() == 0);
    for (std::size_t size : {1U, 3U, 16U, 65U, 1025U}) {
        AlignedBuffer buffer(size);
        CHECK(buffer.size() == size);
        CHECK(reinterpret_cast<std::uintptr_t>(buffer.span().data()) % alignment == 0);
        for (float value : buffer.span()) {
            CHECK(value == 0.0F);
        }
        buffer.span()[size - 1] = 7.0F;
        auto *address = buffer.span().data();
        AlignedBuffer moved(std::move(buffer));
        CHECK(buffer.size() == 0);
        CHECK(buffer.span().data() == nullptr);
        CHECK(moved.span().data() == address);
        AlignedBuffer assigned(2);
        assigned = std::move(moved);
        CHECK(moved.size() == 0);
        CHECK(assigned.span()[size - 1] == 7.0F);
    }
    test::throws<std::length_error>([] { AlignedBuffer bad(max_elements + 1); });
    test::throws<std::length_error>(
        [] { AlignedBuffer bad(std::numeric_limits<std::size_t>::max()); });
    CHECK(std::string_view(error_name(Error::ok)) == "ok");
}

void dense() {
    const std::array<float, 6> weights{1, 2, 3, -1, 0, 2};
    const std::array<float, 2> bias{0.5F, -0.5F};
    const std::array<float, 3> input{2, -1, 0.5F};
    for (auto kernel : {DenseKernel::reference, DenseKernel::packed}) {
        Dense model(3, 2, weights, bias, kernel);
        std::array<float, 2> output{123, 123}, scratch{};
        CHECK(model.forward(input, output, scratch) == Error::ok);
        test::near(output[0], 2);
        test::near(output[1], -1.5);
        CHECK(model.inputs() == 3 && model.outputs() == 2);
        output.fill(123);
        CHECK(model.forward(std::span(input).first(2), output, scratch) == Error::shape_mismatch);
        test::unchanged(output);
        CHECK(model.forward(input, output, std::span(scratch).first(1)) ==
              Error::insufficient_scratch);
        test::unchanged(output);
        auto bad = input;
        bad[1] = nan;
        CHECK(model.forward(bad, output, scratch) == Error::non_finite);
        test::unchanged(output);
        bad[1] = inf;
        CHECK(model.forward(bad, output, scratch) == Error::non_finite);
        test::unchanged(output);
        std::array<float, 6> overlap{1, 2, 3, 123, 123, 123};
        const auto old = overlap;
        CHECK(model.forward(std::span(overlap).first(3), std::span(overlap).subspan(2, 2),
                            scratch) == Error::aliasing);
        CHECK(overlap == old);
        CHECK(model.forward(input, output, output) == Error::aliasing);
        test::unchanged(output);
        auto moved = std::move(model);
        CHECK(model.forward(input, output, scratch) == Error::invalid_state);
        CHECK(moved.forward(input, output, scratch) == Error::ok);
    }
    test::throws<std::invalid_argument>([&] { Dense bad(0, 2, weights, bias); });
    test::throws<std::invalid_argument>([&] { Dense bad(3, 3, weights, bias); });
    test::throws<std::length_error>(
        [&] { Dense bad(std::numeric_limits<std::size_t>::max(), 2, weights, bias); });
    test::throws<std::length_error>([&] { Dense bad(max_elements, 2, weights, bias); });
    auto bad_weights = weights;
    bad_weights[0] = nan;
    test::throws<std::invalid_argument>([&] { Dense bad(3, 2, bad_weights, bias); });
    test::throws<std::invalid_argument>(
        [&] { Dense bad(3, 2, weights, bias, static_cast<DenseKernel>(99)); });
    const std::array<float, 1> w{largest}, b{0}, x{2};
    for (auto kernel : {DenseKernel::reference, DenseKernel::packed}) {
        Dense overflow(1, 1, w, b, kernel);
        std::array<float, 1> y{123}, scratch{};
        CHECK(overflow.forward(x, y, scratch) == Error::numerical_error);
        test::unchanged(y);
    }
    // Construction owns its parameters; caller edits cannot change the model.
    std::array<float, 1> owned_w{3}, owned_b{1};
    Dense owned(1, 1, owned_w, owned_b);
    owned_w[0] = nan;
    owned_b[0] = inf;
    std::array<float, 1> y{}, scratch{};
    CHECK(owned.forward(x, y, scratch) == Error::ok);
    test::near(y[0], 7);
}

void dense_generated() {
    test::Random random;
    for (std::size_t inputs : {1U, 3U, 7U, 8U, 17U, 64U, 256U}) {
        for (std::size_t outputs = 1; outputs <= 19; ++outputs) {
            std::vector<float> w(inputs * outputs), bias(outputs), x(inputs), a(outputs),
                b(outputs), scratch(outputs);
            for (auto &value : w) {
                value = random.next();
            }
            for (auto &value : bias) {
                value = random.next();
            }
            for (auto &value : x) {
                value = random.next();
            }
            Dense reference(inputs, outputs, w, bias, DenseKernel::reference);
            Dense packed(inputs, outputs, w, bias, DenseKernel::packed);
            CHECK(reference.forward(x, a, scratch) == Error::ok);
            CHECK(packed.forward(x, b, scratch) == Error::ok);
            for (std::size_t o = 0; o < outputs; ++o) {
                double expected = bias[o];
                for (std::size_t i = 0; i < inputs; ++i) {
                    expected += static_cast<double>(w[o * inputs + i]) * x[i];
                }
                test::near(a[o], expected, 2e-5, 5e-5);
                test::near(b[o], expected, 2e-5, 5e-5);
                test::near(a[o], b[o], 2e-5, 5e-5);
            }
        }
    }
}

void softmax_cases() {
    std::array<float, 4> input{1, 2, 3, 1000}, output{}, scratch{};
    CHECK(softmax(input, 3, output, scratch) == Error::ok);
    test::near(output[0], 0.09003057317038046);
    test::near(output[1], 0.24472847105479764);
    test::near(output[2], 0.6652409557748219);
    CHECK(output[3] == 0);
    auto expected = output;
    for (auto &x : input) {
        x += 1024;
    }
    CHECK(softmax(input, 3, output, scratch) == Error::ok);
    for (std::size_t i = 0; i < output.size(); ++i) {
        test::near(output[i], expected[i]);
    }
    input.fill(largest);
    CHECK(softmax(input, 4, output, scratch) == Error::ok);
    for (auto value : output) {
        test::near(value, 0.25);
    }
    input = {-largest, largest, 0, 0};
    CHECK(softmax(input, 4, output, scratch) == Error::ok);
    CHECK(output[0] == 0 && output[1] == 1);
    output.fill(123);
    CHECK(softmax(input, 0, output, scratch) == Error::shape_mismatch);
    CHECK(softmax(input, 5, output, scratch) == Error::shape_mismatch);
    test::unchanged(output);
    input[3] = nan;
    CHECK(softmax(input, 1, output, scratch) == Error::non_finite);
    test::unchanged(output);
    CHECK(softmax({}, 0, {}, {}) == Error::shape_mismatch);
    input = {1, 2, 3, 4};
    CHECK(softmax(input, 4, input, scratch) == Error::aliasing);
    CHECK(input[0] == 1);
    CHECK(softmax(input, 4, output, output) == Error::aliasing);
    CHECK(softmax(input, 4, output, std::span(scratch).first(3)) == Error::insufficient_scratch);
    test::unchanged(output);
}

void softmax_generated() {
    test::Random random;
    for (std::size_t size = 1; size <= 127; ++size) {
        std::vector<float> x(size), y(size), scratch(size);
        for (auto &value : x) {
            value = random.next() * 30;
        }
        const auto active = (size + 1) / 2;
        CHECK(softmax(x, active, y, scratch) == Error::ok);
        const double maximum =
            *std::max_element(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(active));
        double total = 0;
        for (std::size_t i = 0; i < active; ++i) {
            total += std::exp(x[i] - maximum);
        }
        double sum = 0;
        for (std::size_t i = 0; i < size; ++i) {
            CHECK(y[i] >= 0 && y[i] <= 1);
            if (i < active) {
                test::near(y[i], std::exp(x[i] - maximum) / total);
            } else {
                CHECK(y[i] == 0);
            }
            sum += y[i];
        }
        test::near(sum, 1);
    }
}

void normalization() {
    std::array<float, 4> x{1, 2, 3, 4}, gamma{1, 2, 0.5F, -1}, beta{0, 1, -1, 2}, y{}, scratch{};
    CHECK(layer_norm(x, gamma, beta, 1e-5F, y, scratch) == Error::ok);
    for (std::size_t i = 0; i < 4; ++i) {
        test::near(y[i], ((x[i] - 2.5) / std::sqrt(1.25 + 1e-5)) * gamma[i] + beta[i]);
    }
    x.fill(largest);
    CHECK(layer_norm(x, gamma, beta, 1e-5F, y, scratch) == Error::ok);
    CHECK(y == beta);
    x = {-largest, largest, -largest, largest};
    gamma.fill(1);
    beta.fill(0);
    CHECK(layer_norm(x, gamma, beta, 1e-5F, y, scratch) == Error::ok);
    test::near(y[0], -1);
    test::near(y[1], 1);
    y.fill(123);
    for (float epsilon : {0.0F, -1.0F, nan, inf}) {
        CHECK(layer_norm(x, gamma, beta, epsilon, y, scratch) == Error::shape_mismatch);
        test::unchanged(y);
    }
    CHECK(layer_norm(x, std::span(gamma).first(3), beta, 1e-5F, y, scratch) ==
          Error::shape_mismatch);
    gamma[0] = nan;
    CHECK(layer_norm(x, gamma, beta, 1e-5F, y, scratch) == Error::non_finite);
    gamma.fill(1);
    x[0] = inf;
    CHECK(layer_norm(x, gamma, beta, 1e-5F, y, scratch) == Error::non_finite);
    test::unchanged(y);
    x = {0, 0, 0, 4};
    gamma.fill(largest);
    CHECK(layer_norm(x, gamma, beta, 1e-5F, y, scratch) == Error::numerical_error);
    test::unchanged(y);
    gamma.fill(1);
    CHECK(layer_norm(x, gamma, beta, 1e-5F, x, scratch) == Error::aliasing);
    CHECK(layer_norm(x, gamma, beta, 1e-5F, y, gamma) == Error::aliasing);
    test::unchanged(y);
}

void activation() {
    std::array<float, 5> x{-3, -1, 0, 1, 3}, y{}, scratch{};
    const std::array<double, 5> expected{-0.0040496940948903, -0.15865525393145707, 0,
                                         0.8413447460685429, 2.99595030590511};
    CHECK(gelu(x, y, scratch) == Error::ok);
    for (std::size_t i = 0; i < x.size(); ++i) {
        test::near(y[i], expected[i]);
    }
    x = {-largest, largest, 0, -10, 10};
    CHECK(gelu(x, y, scratch) == Error::ok);
    CHECK(y[0] == 0 && y[1] == largest && y[2] == 0);
    y.fill(123);
    x[4] = nan;
    CHECK(gelu(x, y, scratch) == Error::non_finite);
    test::unchanged(y);
    x[4] = 10;
    CHECK(gelu(x, x, scratch) == Error::aliasing);
    CHECK(gelu(x, y, x) == Error::aliasing);
    CHECK(gelu(x, std::span(y).first(4), scratch) == Error::shape_mismatch);
    CHECK(gelu(x, y, std::span(scratch).first(4)) == Error::insufficient_scratch);
    CHECK(gelu({}, {}, {}) == Error::shape_mismatch);
    test::unchanged(y);
    test::Random random;
    for (int trial = 0; trial < 200; ++trial) {
        for (auto &value : x) {
            value = random.next() * 4;
        }
        CHECK(gelu(x, y, scratch) == Error::ok);
        for (std::size_t i = 0; i < x.size(); ++i) {
            test::near(y[i],
                       static_cast<double>(x[i]) * 0.5 * (1 + std::erf(x[i] / std::sqrt(2.0))));
        }
    }
}

void residuals() {
    std::array<float, 3> x{1, 2, 3}, r{-1, 5, -0.5F}, y{}, scratch{};
    CHECK(residual_add(x, r, y, scratch) == Error::ok);
    CHECK(y[0] == 0 && y[1] == 7 && y[2] == 2.5F);
    y.fill(123);
    r[2] = nan;
    CHECK(residual_add(x, r, y, scratch) == Error::non_finite);
    test::unchanged(y);
    r[2] = largest;
    x[2] = largest;
    CHECK(residual_add(x, r, y, scratch) == Error::numerical_error);
    test::unchanged(y);
    CHECK(residual_add(x, std::span(r).first(2), y, scratch) == Error::shape_mismatch);
    CHECK(residual_add(x, r, r, scratch) == Error::aliasing);
    CHECK(residual_add(x, r, y, r) == Error::aliasing);
    test::unchanged(y);
    x[2] = -largest;
    CHECK(residual_add(x, r, y, scratch) == Error::ok);
    CHECK(y[2] == 0);
}

void finite_scans() {
    // Every position around vector and tail boundaries must be examined. Include
    // signed zero, subnormal and maximum finite values so a faster predicate
    // cannot accidentally narrow the accepted FP32 domain.
    constexpr std::array<float, 7> valid{0.0F, -0.0F, 1.0F, -1.0F, largest,
        -largest, std::numeric_limits<float>::denorm_min()};
    constexpr std::array<float, 5> invalid{nan, inf, -inf,
        std::bit_cast<float>(std::uint32_t{0xffc00001}),
        std::bit_cast<float>(std::uint32_t{0x7f800001})};
    for (std::size_t size : {1U, 3U, 4U, 7U, 8U, 9U, 15U, 16U, 17U, 63U, 64U, 65U}) {
        std::vector<float> input(size), zero(size, 0.0F), out(size), scratch(size);
        for (std::size_t i = 0; i < size; ++i)
            input[i] = valid[i % valid.size()];
        CHECK(residual_add(input, zero, out, scratch) == Error::ok);
        for (std::size_t i = 0; i < size; ++i)
            CHECK(out[i] == input[i]);
        for (std::size_t i = 0; i < size; ++i) {
            const auto saved = input[i];
            for (float bad : invalid) {
                input[i] = bad;
                std::fill(out.begin(), out.end(), 123.0F);
                CHECK(residual_add(input, zero, out, scratch) == Error::non_finite);
                test::unchanged(out);
                CHECK(residual_add(zero, input, out, scratch) == Error::non_finite);
                test::unchanged(out);
            }
            input[i] = saved;
        }
    }
}
void overlap_boundaries() {
    // Exact adjacency is legal; partial overlap in either address direction is not.
    std::array<float, 12> memory{};
    std::fill(memory.begin(), memory.end(), 1.0F);
    auto input = std::span(memory).first(4);
    auto output = std::span(memory).subspan(4, 4);
    auto scratch = std::span(memory).subspan(8, 4);
    CHECK(gelu(input, output, scratch) == Error::ok);
    const auto before = memory;
    CHECK(gelu(input, std::span(memory).subspan(3, 4), scratch) == Error::aliasing);
    CHECK(memory == before);
    CHECK(gelu(std::span(memory).subspan(3, 4), input, scratch) == Error::aliasing);
    CHECK(memory == before);
    CHECK(gelu(input, output, std::span(memory).subspan(7, 4)) == Error::aliasing);
    CHECK(memory == before);
}
} // namespace
int main(int argc, char *argv[]) {
    try {
        if (argc != 2) {
            throw std::runtime_error("Expected test group");
        }
        const std::string_view group(argv[1]);
        if (group == "buffer")
            buffers();
        else if (group == "dense")
            dense();
        else if (group == "dense_generated")
            dense_generated();
        else if (group == "softmax")
            softmax_cases();
        else if (group == "softmax_generated")
            softmax_generated();
        else if (group == "normalization")
            normalization();
        else if (group == "activation")
            activation();
        else if (group == "residual")
            residuals();
        else if (group == "finite")
            finite_scans();
        else if (group == "overlap")
            overlap_boundaries();
        else
            throw std::runtime_error("Unknown group");
        std::cout << group << ": " << test::checks << " checks passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
