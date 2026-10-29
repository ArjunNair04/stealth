#pragma once
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace test {
inline std::size_t checks = 0;
inline void require(bool condition, const char *expression, int line) {
    ++checks;
    if (!condition) {
        throw std::runtime_error(std::string(expression) + " at line " + std::to_string(line));
    }
}
inline void near(double actual, double expected, double absolute = 2e-6, double relative = 2e-5) {
    ++checks;
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > absolute + relative * std::abs(expected)) {
        throw std::runtime_error("actual=" + std::to_string(actual) +
                                 " expected=" + std::to_string(expected));
    }
}
template <typename Exception, typename Function> void throws(Function &&function) {
    ++checks;
    try {
        std::forward<Function>(function)();
    } catch (const Exception &) {
        return;
    }
    throw std::runtime_error("Expected exception was not thrown");
}
inline void unchanged(std::span<const float> values, float sentinel = 123.0F) {
    for (float value : values) {
        near(value, sentinel, 0, 0);
    }
}
struct Random {
    std::uint32_t state = 0x18273645U;
    float next() {
        state ^= state << 13U;
        state ^= state >> 17U;
        state ^= state << 5U;
        return static_cast<float>(static_cast<int>(state % 257U) - 128) / 64.0F;
    }
};
} // namespace test
#define CHECK(expression) ::test::require(static_cast<bool>(expression), #expression, __LINE__)
