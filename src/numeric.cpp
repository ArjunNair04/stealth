#include "stealth/numeric.hpp"
#include "dense_avx.hpp"
#include "finite.hpp"
#include "fp_policy.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace stealth::numeric {
namespace {
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "Stealth requires IEEE-754 FP32");
constexpr std::size_t lanes = 8;

// Full tiles have fixed-size loads/stores; only the final partial tile needs
// a runtime bound. Keep four named vectors so SLP can vectorize across columns
// without changing the ordered FP32 accumulation along each dot product.
template <std::size_t Rows, bool Full>
void packed_tile(const float *input, const float *weights, const float *bias, float *output,
                 std::size_t inputs, std::size_t outputs, std::size_t count = lanes) noexcept {
    static_assert(Rows == 1 || Rows == 4);
    const auto active = Full ? lanes : count;
    std::array<float, lanes> a{}, b{}, c{}, d{};
    for (std::size_t lane = 0; lane < active; ++lane) {
        a[lane] = bias[lane];
        if constexpr (Rows == 4)
            b[lane] = c[lane] = d[lane] = bias[lane];
    }
    for (std::size_t i = 0; i < inputs; ++i) {
        const float x0 = input[i];
        if constexpr (Rows == 4) {
            const float x1 = input[inputs + i];
            const float x2 = input[2 * inputs + i];
            const float x3 = input[3 * inputs + i];
            for (std::size_t lane = 0; lane < lanes; ++lane) {
                const float weight = weights[i * lanes + lane];
                a[lane] += weight * x0;
                b[lane] += weight * x1;
                c[lane] += weight * x2;
                d[lane] += weight * x3;
            }
        } else {
            for (std::size_t lane = 0; lane < lanes; ++lane)
                a[lane] += weights[i * lanes + lane] * x0;
        }
    }
    for (std::size_t lane = 0; lane < active; ++lane) {
        output[lane] = a[lane];
        if constexpr (Rows == 4) {
            output[outputs + lane] = b[lane];
            output[2 * outputs + lane] = c[lane];
            output[3 * outputs + lane] = d[lane];
        }
    }
}

std::size_t product(std::size_t rows, std::size_t columns) {
    if (rows == 0 || columns == 0) {
        throw std::invalid_argument("Dimensions must be positive");
    }
    if (rows > max_elements || columns > max_elements || rows > max_elements / columns) {
        throw std::length_error("Numeric capacity exceeded");
    }
    return rows * columns;
}

using numeric_detail::finite;

bool overlaps(std::span<const float> a, std::span<const float> b) noexcept {
    if (a.empty() || b.empty())
        return false;
    const auto first = reinterpret_cast<std::uintptr_t>(a.data());
    const auto second = reinterpret_cast<std::uintptr_t>(b.data());
    return first <= second ? second - first < a.size_bytes() : first - second < b.size_bytes();
}

Error validate(std::initializer_list<std::span<const float>> inputs, std::span<float> output,
               std::span<float> scratch) noexcept {
    if (output.empty() || output.size() > max_elements || scratch.size() > max_elements) {
        return Error::shape_mismatch;
    }
    if (scratch.size() < output.size())
        return Error::insufficient_scratch;
    if (overlaps(output, scratch))
        return Error::aliasing;
    for (auto input : inputs) {
        if (input.empty() || input.size() > max_elements)
            return Error::shape_mismatch;
        if (overlaps(input, output) || overlaps(input, scratch))
            return Error::aliasing;
    }
    for (auto input : inputs) {
        if (!finite(input))
            return Error::non_finite;
    }
    return Error::ok;
}

bool checked_float(double value, float &destination) noexcept {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())
        return false;
    destination = static_cast<float>(value);
    return true;
}

Error publish(std::span<const float> scratch, std::span<float> output) noexcept {
    if (!finite(scratch.first(output.size())))
        return Error::numerical_error;
    std::copy_n(scratch.begin(), output.size(), output.begin());
    return Error::ok;
}
} // namespace

const char *error_name(Error error) noexcept {
    switch (error) {
    case Error::ok:
        return "ok";
    case Error::shape_mismatch:
        return "shape_mismatch";
    case Error::non_finite:
        return "non_finite";
    case Error::aliasing:
        return "aliasing";
    case Error::insufficient_scratch:
        return "insufficient_scratch";
    case Error::numerical_error:
        return "numerical_error";
    case Error::invalid_state:
        return "invalid_state";
    }
    return "unknown_error";
}

AlignedBuffer::AlignedBuffer(std::size_t elements) {
    if (elements > max_elements)
        throw std::length_error("Buffer capacity exceeded");
    if (elements != 0) {
        data_ = static_cast<float *>(
            ::operator new[](elements * sizeof(float), std::align_val_t{alignment}));
        size_ = elements;
        std::fill_n(data_, size_, 0.0F);
    }
}
AlignedBuffer::~AlignedBuffer() { ::operator delete[](data_, std::align_val_t{alignment}); }
AlignedBuffer::AlignedBuffer(AlignedBuffer &&other) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}
AlignedBuffer &AlignedBuffer::operator=(AlignedBuffer &&other) noexcept {
    if (this != &other) {
        ::operator delete[](data_, std::align_val_t{alignment});
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
    }
    return *this;
}

Dense::Dense(std::size_t inputs, std::size_t outputs, std::span<const float> weights,
             std::span<const float> bias, DenseKernel kernel)
    : inputs_(inputs), outputs_(outputs), kernel_(kernel) {
    const auto count = product(inputs, outputs);
    if (weights.size() != count || bias.size() != outputs ||
        (kernel != DenseKernel::reference && kernel != DenseKernel::packed)) {
        throw std::invalid_argument("Invalid dense shape or kernel");
    }
    if (!finite(weights) || !finite(bias))
        throw std::invalid_argument("Non-finite parameters");
    const auto padded_outputs = ((outputs + lanes - 1) / lanes) * lanes;
    const auto storage = kernel == DenseKernel::packed ? product(inputs, padded_outputs) : count;
    weights_ = AlignedBuffer(storage);
    bias_ = AlignedBuffer(outputs);
    std::copy(bias.begin(), bias.end(), bias_.span().begin());
    if (kernel == DenseKernel::reference) {
        std::copy(weights.begin(), weights.end(), weights_.span().begin());
    } else {
        for (std::size_t output = 0; output < outputs; ++output) {
            const auto tile = output / lanes;
            const auto lane = output % lanes;
            for (std::size_t input = 0; input < inputs; ++input) {
                weights_.span()[(tile * inputs + input) * lanes + lane] =
                    weights[output * inputs + input];
            }
        }
    }
}

std::size_t Dense::parameter_bytes() const noexcept {
    return (weights_.size() + bias_.size()) * sizeof(float);
}

Error Dense::forward(std::span<const float> input, std::span<float> output,
                     std::span<float> scratch) const noexcept {
    return forward_rows(input, 1, output, scratch);
}

Error Dense::forward_rows(std::span<const float> input, std::size_t rows, std::span<float> output,
                          std::span<float> scratch) const noexcept {
    if (inputs_ == 0 || outputs_ == 0 || bias_.size() != outputs_ || weights_.size() == 0)
        return Error::invalid_state;
    if (rows == 0 || rows > max_elements / inputs_ || rows > max_elements / outputs_ ||
        input.size() != rows * inputs_ || output.size() != rows * outputs_)
        return Error::shape_mismatch;
    const auto status = validate({input}, output, scratch);
    if (status != Error::ok)
        return status;
    const auto weights = weights_.span();
    const auto bias = bias_.span();
    if (kernel_ == DenseKernel::reference) {
        for (std::size_t row = 0; row < rows; ++row)
            for (std::size_t o = 0; o < outputs_; ++o) {
                float accumulator = bias[o];
                for (std::size_t i = 0; i < inputs_; ++i)
                    accumulator += weights[o * inputs_ + i] * input[row * inputs_ + i];
                scratch[row * outputs_ + o] = accumulator;
            }
    } else if (outputs_ >= lanes && (outputs_ >= 2 * lanes || rows >= 4) &&
               numeric_detail::try_dense_avx(input.data(), weights.data(), bias.data(),
                                             scratch.data(), inputs_, outputs_, rows)) {
        // Keep narrow single-row products on the portable path: one AVX
        // accumulator offers little instruction parallelism for the dispatch cost.
        // The specialization writes only disposable scratch. Validation and
        // transactional publication remain common to every backend.
    } else {
        const auto full_outputs = (outputs_ / lanes) * lanes;
        std::size_t row = 0;
        for (; row + 4 <= rows; row += 4) {
            std::size_t start = 0;
            for (; start < full_outputs; start += lanes)
                packed_tile<4, true>(input.data() + row * inputs_, weights.data() + start * inputs_,
                                     bias.data() + start, scratch.data() + row * outputs_ + start,
                                     inputs_, outputs_);
            if (start < outputs_)
                packed_tile<4, false>(input.data() + row * inputs_,
                                      weights.data() + start * inputs_, bias.data() + start,
                                      scratch.data() + row * outputs_ + start, inputs_, outputs_,
                                      outputs_ - start);
        }
        for (; row < rows; ++row) {
            std::size_t start = 0;
            for (; start < full_outputs; start += lanes)
                packed_tile<1, true>(input.data() + row * inputs_, weights.data() + start * inputs_,
                                     bias.data() + start, scratch.data() + row * outputs_ + start,
                                     inputs_, outputs_);
            if (start < outputs_)
                packed_tile<1, false>(input.data() + row * inputs_,
                                      weights.data() + start * inputs_, bias.data() + start,
                                      scratch.data() + row * outputs_ + start, inputs_, outputs_,
                                      outputs_ - start);
        }
    }
    return publish(scratch, output);
}

Error softmax(std::span<const float> input, std::size_t active, std::span<float> output,
              std::span<float> scratch) noexcept {
    if (input.size() != output.size() || active == 0 || active > input.size())
        return Error::shape_mismatch;
    const auto status = validate({input}, output, scratch);
    if (status != Error::ok)
        return status;
    const double maximum =
        *std::max_element(input.begin(), input.begin() + static_cast<std::ptrdiff_t>(active));
    double total = 0;
    for (std::size_t i = 0; i < active; ++i)
        total += std::exp(static_cast<double>(input[i]) - maximum);
    for (std::size_t i = 0; i < active; ++i) {
        scratch[i] = static_cast<float>(std::exp(static_cast<double>(input[i]) - maximum) / total);
    }
    std::fill(scratch.begin() + static_cast<std::ptrdiff_t>(active),
              scratch.begin() + static_cast<std::ptrdiff_t>(output.size()), 0.0F);
    return publish(scratch, output);
}

Error layer_norm(std::span<const float> input, std::span<const float> scale,
                 std::span<const float> shift, float epsilon, std::span<float> output,
                 std::span<float> scratch) noexcept {
    if (input.size() != output.size() || scale.size() != input.size() ||
        shift.size() != input.size() || !std::isfinite(epsilon) || epsilon <= 0)
        return Error::shape_mismatch;
    const auto status = validate({input, scale, shift}, output, scratch);
    if (status != Error::ok)
        return status;
    double sum = 0;
    for (float value : input)
        sum += value;
    const double mean = sum / static_cast<double>(input.size());
    double variance = 0;
    for (float value : input) {
        const double difference = value - mean;
        variance += difference * difference;
    }
    variance /= static_cast<double>(input.size());
    const double inverse = 1.0 / std::sqrt(variance + epsilon);
    bool representable = true;
    for (std::size_t i = 0; i < input.size(); ++i) {
        const double value = ((input[i] - mean) * inverse) * scale[i] + shift[i];
        const bool valid = std::abs(value) <= std::numeric_limits<float>::max();
        // Mask before narrowing: out-of-range/NaN values are never converted.
        // Scratch may change on failure, but output is published only if every
        // channel passes. The reduction lets independent channels vectorize.
        scratch[i] = static_cast<float>(valid ? value : 0.0);
        representable &= valid;
    }
    if (!representable)
        return Error::numerical_error;
    return publish(scratch, output);
}

Error layer_norm_rows(std::span<const float> input, std::size_t rows, std::span<const float> scale,
                      std::span<const float> shift, float epsilon, std::span<float> output,
                      std::span<float> scratch) noexcept {
    if (rows == 1)
        return layer_norm(input, scale, shift, epsilon, output, scratch);
    const auto width = scale.size();
    if (rows == 0 || width == 0 || width > max_elements || rows > max_elements / width ||
        shift.size() != width || input.size() != rows * width || output.size() != input.size() ||
        !std::isfinite(epsilon) || epsilon <= 0)
        return Error::shape_mismatch;
    const auto status = validate({input, scale, shift}, output, scratch);
    if (status != Error::ok)
        return status;
    // Interleave independent reductions so the compiler can vectorize across
    // rows. Never reassociate the sum or variance within a row.
    const auto tile = [&]<std::size_t Rows>(std::size_t first) {
        std::array<double, Rows> means{}, variances{}, inverses{};
        for (std::size_t i = 0; i < width; ++i)
            for (std::size_t r = 0; r < Rows; ++r)
                means[r] += input[(first + r) * width + i];
        for (auto &mean : means)
            mean /= static_cast<double>(width);
        for (std::size_t i = 0; i < width; ++i)
            for (std::size_t r = 0; r < Rows; ++r) {
                const double difference = input[(first + r) * width + i] - means[r];
                variances[r] += difference * difference;
            }
        for (std::size_t r = 0; r < Rows; ++r) {
            variances[r] /= static_cast<double>(width);
            inverses[r] = 1.0 / std::sqrt(variances[r] + epsilon);
        }
        bool representable = true;
        for (std::size_t r = 0; r < Rows; ++r)
            for (std::size_t i = 0; i < width; ++i) {
                const auto index = (first + r) * width + i;
                const double value =
                    ((input[index] - means[r]) * inverses[r]) * scale[i] + shift[i];
                const bool valid = std::abs(value) <= std::numeric_limits<float>::max();
                scratch[index] = static_cast<float>(valid ? value : 0.0);
                representable &= valid;
            }
        return representable;
    };
    std::size_t row = 0;
    for (; row + 4 <= rows; row += 4)
        if (!tile.template operator()<4>(row))
            return Error::numerical_error;
    for (; row < rows; ++row)
        if (!tile.template operator()<1>(row))
            return Error::numerical_error;
    return publish(scratch, output);
}

Error gelu(std::span<const float> input, std::span<float> output,
           std::span<float> scratch) noexcept {
    if (input.size() != output.size())
        return Error::shape_mismatch;
    const auto status = validate({input}, output, scratch);
    if (status != Error::ok)
        return status;
    constexpr double inverse_sqrt_two = 0.7071067811865475244;
    for (std::size_t i = 0; i < input.size(); ++i) {
        const double value =
            0.5 * input[i] * std::erfc(-static_cast<double>(input[i]) * inverse_sqrt_two);
        if (!checked_float(value, scratch[i]))
            return Error::numerical_error;
    }
    return publish(scratch, output);
}

Error residual_add(std::span<const float> input, std::span<const float> residual,
                   std::span<float> output, std::span<float> scratch) noexcept {
    if (input.size() != output.size() || residual.size() != input.size())
        return Error::shape_mismatch;
    const auto status = validate({input, residual}, output, scratch);
    if (status != Error::ok)
        return status;
    for (std::size_t i = 0; i < input.size(); ++i) {
        if (!checked_float(static_cast<double>(input[i]) + residual[i], scratch[i]))
            return Error::numerical_error;
    }
    return publish(scratch, output);
}
} // namespace stealth::numeric
