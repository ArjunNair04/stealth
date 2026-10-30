#include "dense_avx.hpp"
#include "fp_policy.hpp"

#if STEALTH_ENABLE_AVX && defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
#define STEALTH_HAS_AVX_TARGET 1
#include <algorithm>
#include <immintrin.h>
#else
#define STEALTH_HAS_AVX_TARGET 0
#endif

namespace stealth::numeric::numeric_detail {
bool dense_avx_available() noexcept {
#if STEALTH_HAS_AVX_TARGET
    // Compiler runtime checks usable AVX, including OS vector-state support.
    return __builtin_cpu_supports("avx") != 0;
#else
    return false;
#endif
}

#if STEALTH_HAS_AVX_TARGET
namespace {
// Only this function family may contain AVX instructions. No -march=native or
// whole-library AVX flag: the dispatcher and portable kernel remain baseline ISA.
template <std::size_t Rows, bool Full>
__attribute__((target("avx")))
void tile(const float *input, const float *weights, const float *bias, float *output,
          std::size_t inputs, std::size_t outputs, std::size_t count = 8) noexcept {
    __m256 a;
    if constexpr (Full) {
        a = _mm256_loadu_ps(bias);
    } else {
        float padded[8]{};
        std::copy_n(bias, count, padded);
        a = _mm256_loadu_ps(padded);
    }
    __m256 b = a, c = a, d = a;
    for (std::size_t i = 0; i < inputs; ++i) {
        const auto weight = _mm256_loadu_ps(weights + i * 8);
        // Explicit independent lanes; the sequence along each dot product is
        // unchanged. The core's -ffp-contract=off also applies to this source.
        a = _mm256_add_ps(a, _mm256_mul_ps(weight, _mm256_set1_ps(input[i])));
        if constexpr (Rows == 4) {
            b = _mm256_add_ps(b, _mm256_mul_ps(weight, _mm256_set1_ps(input[inputs + i])));
            c = _mm256_add_ps(c, _mm256_mul_ps(weight, _mm256_set1_ps(input[2 * inputs + i])));
            d = _mm256_add_ps(d, _mm256_mul_ps(weight, _mm256_set1_ps(input[3 * inputs + i])));
        }
    }
    if constexpr (Full) {
        _mm256_storeu_ps(output, a);
        if constexpr (Rows == 4) {
            _mm256_storeu_ps(output + outputs, b);
            _mm256_storeu_ps(output + 2 * outputs, c);
            _mm256_storeu_ps(output + 3 * outputs, d);
        }
    } else {
        float tail[8];
        _mm256_storeu_ps(tail, a);
        std::copy_n(tail, count, output);
        if constexpr (Rows == 4) {
            _mm256_storeu_ps(tail, b);
            std::copy_n(tail, count, output + outputs);
            _mm256_storeu_ps(tail, c);
            std::copy_n(tail, count, output + 2 * outputs);
            _mm256_storeu_ps(tail, d);
            std::copy_n(tail, count, output + 3 * outputs);
        }
    }
}
__attribute__((target("avx")))
void forward(const float *input, const float *weights, const float *bias, float *scratch,
             std::size_t inputs, std::size_t outputs, std::size_t rows) noexcept {
    const auto full = (outputs / 8) * 8;
    std::size_t row = 0;
    for (; row + 4 <= rows; row += 4) {
        std::size_t start = 0;
        for (; start < full; start += 8)
            tile<4, true>(input + row * inputs, weights + start * inputs, bias + start,
                          scratch + row * outputs + start, inputs, outputs);
        if (start < outputs)
            tile<4, false>(input + row * inputs, weights + start * inputs, bias + start,
                           scratch + row * outputs + start, inputs, outputs, outputs - start);
    }
    for (; row < rows; ++row) {
        std::size_t start = 0;
        for (; start < full; start += 8)
            tile<1, true>(input + row * inputs, weights + start * inputs, bias + start,
                          scratch + row * outputs + start, inputs, outputs);
        if (start < outputs)
            tile<1, false>(input + row * inputs, weights + start * inputs, bias + start,
                           scratch + row * outputs + start, inputs, outputs, outputs - start);
    }
}
} // namespace
#endif

bool try_dense_avx(const float *input, const float *weights, const float *bias, float *scratch,
                   std::size_t inputs, std::size_t outputs, std::size_t rows) noexcept {
#if STEALTH_HAS_AVX_TARGET
    if (dense_avx_available()) {
        forward(input, weights, bias, scratch, inputs, outputs, rows);
        return true;
    }
#else
    (void)input; (void)weights; (void)bias; (void)scratch;
    (void)inputs; (void)outputs; (void)rows;
#endif
    return false;
}
} // namespace stealth::numeric::numeric_detail
