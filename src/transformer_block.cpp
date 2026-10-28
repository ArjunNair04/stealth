#include "stealth/transformer_block.hpp"
#include "fp_policy.hpp"
#include "sequence_detail.hpp"
#include <algorithm>

namespace stealth::numeric {
namespace {
std::size_t checked_width(std::size_t width, std::size_t heads) {
    sequence_detail::dimensions(1, width, heads);
    return width;
}
std::size_t checked_hidden(std::size_t hidden) {
    if (hidden == 0)
        throw std::invalid_argument("Hidden width must be positive");
    if (hidden > max_hidden)
        throw std::length_error("Hidden width capacity exceeded");
    return hidden;
}
float checked_epsilon(float epsilon) {
    if (!std::isfinite(epsilon) || epsilon <= 0)
        throw std::invalid_argument("Epsilon must be finite and positive");
    return epsilon;
}
std::size_t workspace_elements(std::size_t capacity, std::size_t width, std::size_t heads,
                               std::size_t hidden) {
    sequence_detail::dimensions(capacity, width, heads);
    (void)checked_hidden(hidden);
    // Five sequences (Q,K,V,context,candidate), six row buffers, and 3*width scratch.
    return (5 * capacity + 9) * width;
}
} // namespace
TransformerWorkspace::TransformerWorkspace(std::size_t capacity, std::size_t width,
                                           std::size_t heads, std::size_t hidden)
    : capacity_(capacity), width_(width), heads_(heads), hidden_(hidden),
      storage_(workspace_elements(capacity, width, heads, hidden)),
      attention_(capacity, width, heads), feed_forward_(width, hidden) {}

TransformerBlock::TransformerBlock(std::size_t width, std::size_t heads, std::size_t hidden,
                                   TransformerParameters p, float epsilon, DenseKernel kernel)
    : width_(checked_width(width, heads)), heads_(heads), hidden_(checked_hidden(hidden)),
      epsilon_(checked_epsilon(epsilon)),
      qkv_(width_, 3 * width_, p.qkv_weights, p.qkv_bias, kernel),
      projection_(width_, width_, p.projection_weights, p.projection_bias, kernel),
      feed_forward_(width_, hidden_, p.feed_forward, epsilon_, kernel), scale_(width_),
      shift_(width_) {
    if (p.attention_scale.size() != width_ || p.attention_shift.size() != width_ ||
        !sequence_detail::finite(p.attention_scale) || !sequence_detail::finite(p.attention_shift))
        throw std::invalid_argument("Invalid attention normalization parameters");
    std::copy(p.attention_scale.begin(), p.attention_scale.end(), scale_.span().begin());
    std::copy(p.attention_shift.begin(), p.attention_shift.end(), shift_.span().begin());
}
std::size_t TransformerBlock::parameter_bytes() const noexcept {
    return qkv_.parameter_bytes() + projection_.parameter_bytes() +
           feed_forward_.parameter_bytes() + (scale_.size() + shift_.size()) * sizeof(float);
}
Error TransformerBlock::forward(std::span<const float> input, std::size_t tokens,
                                std::span<float> output,
                                TransformerWorkspace &workspace) const noexcept {
    if (scale_.size() != width_ || shift_.size() != width_ || workspace.storage_.size() == 0)
        return Error::invalid_state;
    if (tokens == 0 || tokens > workspace.capacity_ || workspace.width_ != width_ ||
        workspace.heads_ != heads_ || workspace.hidden_ != hidden_)
        return Error::shape_mismatch;
    const auto count = tokens * width_;
    if (input.size() != count || output.size() != count)
        return Error::shape_mismatch;
    if (sequence_detail::overlaps(input, output))
        return Error::aliasing;
    if (!sequence_detail::finite(input))
        return Error::non_finite;
    const auto capacity_count = workspace.capacity_ * width_;
    auto storage = workspace.storage_.span();
    auto query = storage.subspan(0, count);
    auto key = storage.subspan(capacity_count, count);
    auto value = storage.subspan(2 * capacity_count, count);
    auto context = storage.subspan(3 * capacity_count, count);
    auto candidate = storage.subspan(4 * capacity_count, count);
    const auto offset = 5 * capacity_count;
    auto normalized = storage.subspan(offset, width_);
    auto qkv = storage.subspan(offset + width_, 3 * width_);
    auto projected = storage.subspan(offset + 4 * width_, width_);
    auto residual = storage.subspan(offset + 5 * width_, width_);
    auto scratch = storage.subspan(offset + 6 * width_, 3 * width_);
    for (std::size_t t = 0; t < tokens; ++t) {
        auto status = layer_norm(input.subspan(t * width_, width_), scale_.span(), shift_.span(),
                                 epsilon_, normalized, scratch);
        if (status != Error::ok)
            return status;
        status = qkv_.forward(normalized, qkv, scratch);
        if (status != Error::ok)
            return status;
        std::copy_n(qkv.begin(), width_, query.begin() + static_cast<std::ptrdiff_t>(t * width_));
        std::copy_n(qkv.begin() + static_cast<std::ptrdiff_t>(width_), width_,
                    key.begin() + static_cast<std::ptrdiff_t>(t * width_));
        std::copy_n(qkv.begin() + static_cast<std::ptrdiff_t>(2 * width_), width_,
                    value.begin() + static_cast<std::ptrdiff_t>(t * width_));
    }
    auto status = causal_attention(query, key, value, tokens, context, workspace.attention_);
    if (status != Error::ok)
        return status;
    for (std::size_t t = 0; t < tokens; ++t) {
        status = projection_.forward(context.subspan(t * width_, width_), projected, scratch);
        if (status != Error::ok)
            return status;
        status = residual_add(input.subspan(t * width_, width_), projected, residual, scratch);
        if (status != Error::ok)
            return status;
        status = feed_forward_.forward(residual, candidate.subspan(t * width_, width_),
                                       workspace.feed_forward_);
        if (status != Error::ok)
            return status;
    }
    std::copy(candidate.begin(), candidate.end(), output.begin());
    return Error::ok;
}
} // namespace stealth::numeric
