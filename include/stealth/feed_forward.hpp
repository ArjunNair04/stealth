#pragma once
#include "stealth/numeric.hpp"

namespace stealth::numeric {
struct FeedForwardParameters {
    std::span<const float> scale;
    std::span<const float> shift;
    std::span<const float> up_weights;
    std::span<const float> up_bias;
    std::span<const float> down_weights;
    std::span<const float> down_bias;
};

class FeedForwardWorkspace {
  public:
    FeedForwardWorkspace(std::size_t width, std::size_t hidden);
    [[nodiscard]] std::size_t bytes() const noexcept { return storage_.size() * sizeof(float); }

  private:
    friend class FeedForward;
    std::size_t width_;
    std::size_t hidden_;
    AlignedBuffer storage_;
};

class FeedForward {
  public:
    FeedForward(std::size_t width, std::size_t hidden, FeedForwardParameters parameters,
                float epsilon = 1e-5F, DenseKernel kernel = DenseKernel::packed);
    [[nodiscard]] Error forward(std::span<const float> input, std::span<float> output,
                                FeedForwardWorkspace &workspace) const noexcept;
    [[nodiscard]] std::size_t parameter_bytes() const noexcept;
    [[nodiscard]] std::size_t width() const noexcept { return width_; }
    [[nodiscard]] std::size_t hidden() const noexcept { return hidden_; }

  private:
    std::size_t width_;
    std::size_t hidden_;
    float epsilon_;
    Dense up_;
    Dense down_;
    AlignedBuffer scale_;
    AlignedBuffer shift_;
};
} // namespace stealth::numeric
