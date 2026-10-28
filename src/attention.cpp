#include "stealth/attention.hpp"
#include "fp_policy.hpp"
#include "sequence_detail.hpp"
#include <algorithm>

namespace stealth::numeric {
namespace {
std::size_t workspace_elements(std::size_t capacity, std::size_t width, std::size_t heads) {
    sequence_detail::dimensions(capacity, width, heads);
    return capacity * (width + 3); // Bounded dimensions validated before multiplication.
}
} // namespace
AttentionWorkspace::AttentionWorkspace(std::size_t capacity, std::size_t width, std::size_t heads)
    : capacity_(capacity), width_(width), heads_(heads),
      storage_(workspace_elements(capacity, width, heads)) {}

Error causal_attention(std::span<const float> query, std::span<const float> key,
                       std::span<const float> value, std::size_t tokens, std::span<float> output,
                       AttentionWorkspace &workspace) noexcept {
    using namespace sequence_detail;
    if (workspace.storage_.size() == 0)
        return Error::invalid_state;
    if (tokens == 0 || tokens > workspace.capacity_)
        return Error::shape_mismatch;
    const auto width = workspace.width_;
    const auto count = tokens * width;
    if (query.size() != count || key.size() != count || value.size() != count ||
        output.size() != count)
        return Error::shape_mismatch;
    if (overlaps(query, output) || overlaps(key, output) || overlaps(value, output))
        return Error::aliasing;
    if (!finite(query) || !finite(key) || !finite(value))
        return Error::non_finite;
    auto storage = workspace.storage_.span();
    auto candidate = storage.first(count);
    const auto offset = workspace.capacity_ * width;
    auto scores = storage.subspan(offset, workspace.capacity_);
    auto probabilities = storage.subspan(offset + workspace.capacity_, workspace.capacity_);
    auto scratch = storage.subspan(offset + 2 * workspace.capacity_, workspace.capacity_);
    const auto head_width = width / workspace.heads_;
    const double inverse_scale = 1.0 / std::sqrt(static_cast<double>(head_width));
    for (std::size_t t = 0; t < tokens; ++t) {
        const auto active = t + 1;
        for (std::size_t head = 0; head < workspace.heads_; ++head) {
            const auto channel = head * head_width;
            for (std::size_t j = 0; j < active; ++j) {
                double dot = 0;
                for (std::size_t c = 0; c < head_width; ++c)
                    dot += static_cast<double>(query[t * width + channel + c]) *
                           key[j * width + channel + c];
                if (!checked_float(dot * inverse_scale, scores[j]))
                    return Error::numerical_error;
            }
            const auto status =
                softmax(scores.first(active), active, probabilities.first(active), scratch);
            if (status != Error::ok)
                return status;
            for (std::size_t c = 0; c < head_width; ++c) {
                double sum = 0;
                for (std::size_t j = 0; j < active; ++j)
                    sum += static_cast<double>(probabilities[j]) * value[j * width + channel + c];
                if (!checked_float(sum, candidate[t * width + channel + c]))
                    return Error::numerical_error;
            }
        }
    }
    std::copy(candidate.begin(), candidate.end(), output.begin());
    return Error::ok;
}
} // namespace stealth::numeric
