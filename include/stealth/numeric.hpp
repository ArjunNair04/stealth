#pragma once

#include <cstddef>
#include <span>

namespace stealth::numeric {

inline constexpr std::size_t alignment = 64;
inline constexpr std::size_t max_elements = 16'777'216;

enum class Error {
    ok,
    shape_mismatch,
    non_finite,
    aliasing,
    insufficient_scratch,
    numerical_error,
    invalid_state
};
[[nodiscard]] const char *error_name(Error error) noexcept;

enum class DenseKernel { reference, packed };

// Move-only ownership. Allocation and zero-initialization occur only here.
class AlignedBuffer {
  public:
    explicit AlignedBuffer(std::size_t elements = 0);
    ~AlignedBuffer();
    AlignedBuffer(const AlignedBuffer &) = delete;
    AlignedBuffer &operator=(const AlignedBuffer &) = delete;
    AlignedBuffer(AlignedBuffer &&other) noexcept;
    AlignedBuffer &operator=(AlignedBuffer &&other) noexcept;
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::span<float> span() noexcept { return {data_, size_}; }
    [[nodiscard]] std::span<const float> span() const noexcept { return {data_, size_}; }

  private:
    float *data_ = nullptr;
    std::size_t size_ = 0;
};

// weights[out,in] and bias[out] are copied, validated and optionally packed once.
class Dense {
  public:
    Dense(std::size_t inputs, std::size_t outputs, std::span<const float> weights,
          std::span<const float> bias, DenseKernel kernel = DenseKernel::packed);
    Dense(const Dense &) = delete;
    Dense &operator=(const Dense &) = delete;
    Dense(Dense &&) noexcept = default;
    Dense &operator=(Dense &&) noexcept = default;
    [[nodiscard]] Error forward(std::span<const float> input, std::span<float> output,
                                std::span<float> scratch) const noexcept;
    // Contiguous rows; scratch stages the complete output before publication.
    [[nodiscard]] Error forward_rows(std::span<const float> input, std::size_t rows,
                                     std::span<float> output,
                                     std::span<float> scratch) const noexcept;
    [[nodiscard]] std::size_t inputs() const noexcept { return inputs_; }
    [[nodiscard]] std::size_t outputs() const noexcept { return outputs_; }
    [[nodiscard]] std::size_t parameter_bytes() const noexcept;

  private:
    std::size_t inputs_ = 0;
    std::size_t outputs_ = 0;
    DenseKernel kernel_ = DenseKernel::reference;
    AlignedBuffer weights_;
    AlignedBuffer bias_;
};

// All hot operations preserve output on failure; scratch is disposable.
// Output is disjoint from inputs, and the entire scratch span is disjoint from all.
[[nodiscard]] Error softmax(std::span<const float> input, std::size_t active,
                            std::span<float> output, std::span<float> scratch) noexcept;
[[nodiscard]] Error layer_norm(std::span<const float> input, std::span<const float> scale,
                               std::span<const float> shift, float epsilon, std::span<float> output,
                               std::span<float> scratch) noexcept;
// Row-wise normalization with shared scale/shift. Each row retains scalar FP64
// reduction order. Scratch stages every row before publishing any output.
[[nodiscard]] Error layer_norm_rows(std::span<const float> input, std::size_t rows,
                                    std::span<const float> scale, std::span<const float> shift,
                                    float epsilon, std::span<float> output,
                                    std::span<float> scratch) noexcept;
[[nodiscard]] Error gelu(std::span<const float> input, std::span<float> output,
                         std::span<float> scratch) noexcept;
[[nodiscard]] Error residual_add(std::span<const float> input, std::span<const float> residual,
                                 std::span<float> output, std::span<float> scratch) noexcept;

} // namespace stealth::numeric
