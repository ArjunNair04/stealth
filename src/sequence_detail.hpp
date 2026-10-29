#pragma once
#include "stealth/attention.hpp"
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace stealth::numeric::sequence_detail {
inline void dimensions(std::size_t capacity, std::size_t width, std::size_t heads) {
    if (capacity == 0 || width == 0 || heads == 0)
        throw std::invalid_argument("Sequence dimensions must be positive");
    if (capacity > max_context || width > max_width)
        throw std::length_error("Sequence capacity exceeded");
    if (heads > width || width % heads != 0)
        throw std::invalid_argument("Heads must divide width");
}
inline bool finite(std::span<const float> values) noexcept {
    for (float value : values)
        if (!std::isfinite(value))
            return false;
    return true;
}
inline bool overlaps(std::span<const float> a, std::span<const float> b) noexcept {
    if (a.empty() || b.empty())
        return false;
    const auto first = reinterpret_cast<std::uintptr_t>(a.data());
    const auto second = reinterpret_cast<std::uintptr_t>(b.data());
    return first <= second ? second - first < a.size_bytes() : first - second < b.size_bytes();
}
inline bool checked_float(double value, float &out) noexcept {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())
        return false;
    out = static_cast<float>(value);
    return true;
}
} // namespace stealth::numeric::sequence_detail
