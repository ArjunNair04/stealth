#pragma once
#include <cstddef>

namespace stealth::numeric::numeric_detail {
// Private specialization. Caller has validated dimensions, disjoint buffers and
// finite input. Weights use the existing padded eight-output packed layout.
// A false return leaves scratch untouched so the caller can use the portable path.
bool dense_avx_available() noexcept;
bool try_dense_avx(const float *input, const float *weights, const float *bias,
                   float *scratch, std::size_t inputs, std::size_t outputs,
                   std::size_t rows) noexcept;
} // namespace stealth::numeric::numeric_detail
