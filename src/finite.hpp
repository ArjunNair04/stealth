#pragma once
#include <cmath>
#include <span>

namespace stealth::numeric::numeric_detail {
// A reduction permits SIMD validation of the entire span. Do not use && here:
// a short-circuit exit inhibits vectorization. Invalid input is still rejected
// before arithmetic or publication; no finite-value checks are removed.
inline bool finite(std::span<const float> values) noexcept {
    bool valid = true;
    for (float value : values)
        valid &= std::isfinite(value);
    return valid;
}
} // namespace stealth::numeric::numeric_detail
