#include "stealth/feed_forward.hpp"
#include "fp_policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace stealth::numeric {
namespace {
std::size_t workspace_elements(std::size_t width, std::size_t hidden, std::size_t capacity = 1) {
    if (width == 0 || hidden == 0 || capacity == 0)
        throw std::invalid_argument("Workspace dimensions must be positive");
    if (width > max_elements || hidden > max_elements)
        throw std::length_error("Workspace capacity exceeded");
    const auto count = 2 * width + 2 * hidden + std::max(width, hidden);
    if (count > max_elements / capacity)
        throw std::length_error("Workspace capacity exceeded");
    return count * capacity;
}

bool overlap(std::span<const float> input, std::span<float> output) noexcept {
    const auto a = reinterpret_cast<std::uintptr_t>(input.data());
    const auto b = reinterpret_cast<std::uintptr_t>(output.data());
    return a <= b ? b - a < input.size_bytes() : a - b < output.size_bytes();
}
} // namespace

FeedForwardWorkspace::FeedForwardWorkspace(std::size_t width, std::size_t hidden,
                                           std::size_t capacity)
    : capacity_(capacity), width_(width), hidden_(hidden),
      storage_(workspace_elements(width, hidden, capacity)) {}

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
    return forward_rows(input, 1, output, workspace);
}
Error FeedForward::forward_rows(std::span<const float> input, std::size_t rows,
                                std::span<float> output,
                                FeedForwardWorkspace &workspace) const noexcept {
    if (scale_.size() != width_ || shift_.size() != width_ || workspace.storage_.size() == 0)
        return Error::invalid_state;
    if (rows == 0 || rows > workspace.capacity_ || workspace.width_ != width_ ||
        workspace.hidden_ != hidden_ || input.size() != rows * width_ ||
        output.size() != rows * width_)
        return Error::shape_mismatch;
    if (overlap(input, output))
        return Error::aliasing;
    for (float value : input)
        if (!std::isfinite(value))
            return Error::non_finite;

    const auto w = rows * width_, h = rows * hidden_;
    auto storage = workspace.storage_.span();
    auto normalized = storage.first(w);
    auto expanded = storage.subspan(w, h);
    auto activated = storage.subspan(w + h, h);
    auto projected = storage.subspan(w + 2 * h, w);
    auto scratch = storage.subspan(2 * w + 2 * h);
    for (std::size_t row = 0; row < rows; ++row) {
        const auto status =
            layer_norm(input.subspan(row * width_, width_), scale_.span(), shift_.span(), epsilon_,
                       normalized.subspan(row * width_, width_), scratch);
        if (status != Error::ok)
            return status;
    }
    auto status = up_.forward_rows(normalized, rows, expanded, scratch);
    if (status != Error::ok)
        return status;
    status = gelu(expanded, activated, scratch);
    if (status != Error::ok)
        return status;
    status = down_.forward_rows(activated, rows, projected, scratch);
    if (status != Error::ok)
        return status;
    // Only this last operation publishes, after every row/stage passes.
    return residual_add(input, projected, output, scratch);
}
} // namespace stealth::numeric
