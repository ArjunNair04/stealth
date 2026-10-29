#pragma once
#include "stealth/attention.hpp"
#include "stealth/feed_forward.hpp"

namespace stealth::numeric {
struct TransformerParameters {
    std::span<const float> attention_scale, attention_shift;
    std::span<const float> qkv_weights, qkv_bias;
    std::span<const float> projection_weights, projection_bias;
    FeedForwardParameters feed_forward;
};
class TransformerWorkspace {
  public:
    TransformerWorkspace(std::size_t capacity, std::size_t width, std::size_t heads,
                         std::size_t hidden);
    [[nodiscard]] std::size_t bytes() const noexcept {
        return storage_.size() * sizeof(float) + attention_.bytes() + feed_forward_.bytes();
    }

  private:
    friend class TransformerBlock;
    std::size_t capacity_, width_, heads_, hidden_;
    AlignedBuffer storage_;
    AttentionWorkspace attention_;
    FeedForwardWorkspace feed_forward_;
};
// A pre-norm block on embedded rows. The caller supplies position encoding.
class TransformerBlock {
  public:
    TransformerBlock(std::size_t width, std::size_t heads, std::size_t hidden,
                     TransformerParameters parameters, float epsilon = 1e-5F,
                     DenseKernel kernel = DenseKernel::packed);
    [[nodiscard]] Error forward(std::span<const float> input, std::size_t tokens,
                                std::span<float> output, TransformerWorkspace &workspace,
                                SequenceOutput mode = SequenceOutput::all) const noexcept;
    [[nodiscard]] std::size_t parameter_bytes() const noexcept;

  private:
    std::size_t width_, heads_, hidden_;
    float epsilon_;
    Dense qkv_, projection_;
    FeedForward feed_forward_;
    AlignedBuffer scale_, shift_;
};
} // namespace stealth::numeric
