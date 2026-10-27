#include "stealth/feed_forward.hpp"
#include "fp_policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace stealth::numeric {
namespace {
std::size_t workspace_elements(std::size_t width, std::size_t hidden) {
    if (width == 0 || hidden == 0)
        throw std::invalid_argument("Workspace dimensions must be positive");
    if (width > max_elements || hidden > max_elements)
        throw std::length_error("Workspace capacity exceeded");
    const auto count = 2 * width + 2 * hidden + std::max(width, hidden);
    if (count > max_elements)
        throw std::length_error("Workspace capacity exceeded");
    return count;
}

bool overlap(std::span<const float> input, std::span<float> output) noexcept {
    const auto a = reinterpret_cast<std::uintptr_t>(input.data());
    const auto b = reinterpret_cast<std::uintptr_t>(output.data());
    return a <= b ? b - a < input.size_bytes() : a - b < output.size_bytes();
}
} // namespace

FeedForwardWorkspace::FeedForwardWorkspace(std::size_t width, std::size_t hidden)
    : width_(width), hidden_(hidden), storage_(workspace_elements(width, hidden)) {}

FeedForward::FeedForward(std::size_t width, std::size_t hidden, FeedForwardParameters parameters,
                         float epsilon, DenseKernel kernel)
    : width_(width), hidden_(hidden), epsilon_(epsilon),
      up_(width, hidden, parameters.up_weights, parameters.up_bias, kernel),
      down_(hidden, width, parameters.down_weights, parameters.down_bias, kernel), scale_(width),
      shift_(width) {
    // Also reject dimensions for which no bounded workspace can be constructed.
    (void)workspace_elements(width, hidden);
    if (parameters.scale.size() != width || parameters.shift.size() != width ||
        !std::isfinite(epsilon) || epsilon <= 0) {
        throw std::invalid_argument("Invalid layer normalization parameters");
    }
    for (std::size_t i = 0; i < width; ++i) {
        if (!std::isfinite(parameters.scale[i]) || !std::isfinite(parameters.shift[i])) {
            throw std::invalid_argument("Non-finite layer normalization parameters");
        }
    }
    std::copy(parameters.scale.begin(), parameters.scale.end(), scale_.span().begin());
    std::copy(parameters.shift.begin(), parameters.shift.end(), shift_.span().begin());
}

std::size_t FeedForward::parameter_bytes() const noexcept {
    return up_.parameter_bytes() + down_.parameter_bytes() +
           (scale_.size() + shift_.size()) * sizeof(float);
}

Error FeedForward::forward(std::span<const float> input, std::span<float> output,
                           FeedForwardWorkspace &workspace) const noexcept {
    if (scale_.size() != width_ || shift_.size() != width_ || workspace.storage_.size() == 0) {
        return Error::invalid_state;
    }
    if (input.size() != width_ || output.size() != width_ || workspace.width_ != width_ ||
        workspace.hidden_ != hidden_)
        return Error::shape_mismatch;
    if (overlap(input, output))
        return Error::aliasing;

    auto storage = workspace.storage_.span();
    auto normalized = storage.first(width_);
    auto expanded = storage.subspan(width_, hidden_);
    auto activated = storage.subspan(width_ + hidden_, hidden_);
    auto projected = storage.subspan(width_ + 2 * hidden_, width_);
    auto scratch = storage.subspan(2 * width_ + 2 * hidden_);

    auto status = layer_norm(input, scale_.span(), shift_.span(), epsilon_, normalized, scratch);
    if (status != Error::ok)
        return status;
    status = up_.forward(normalized, expanded, scratch);
    if (status != Error::ok)
        return status;
    status = gelu(expanded, activated, scratch);
    if (status != Error::ok)
        return status;
    status = down_.forward(activated, projected, scratch);
    if (status != Error::ok)
        return status;
    // Only this last operation publishes to caller-owned output, after all stages pass.
    return residual_add(input, projected, output, scratch);
}
} // namespace stealth::numeric
