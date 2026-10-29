#include "stealth/attention.hpp"
#include "fp_policy.hpp"
#include "sequence_detail.hpp"
#include <algorithm>
#include <array>

namespace stealth::numeric {
namespace {
std::size_t workspace_elements(std::size_t capacity, std::size_t width, std::size_t heads,
                               AttentionKernel kernel) {
    if (kernel != AttentionKernel::reference && kernel != AttentionKernel::tiled)
        throw std::invalid_argument("Invalid attention kernel");
    sequence_detail::dimensions(capacity, width, heads);
    return capacity * (width + 3); // Bounded dimensions validated before multiplication.
}
} // namespace
AttentionWorkspace::AttentionWorkspace(std::size_t capacity, std::size_t width, std::size_t heads,
                                       AttentionKernel kernel)
    : capacity_(capacity), width_(width), heads_(heads), kernel_(kernel),
      storage_(workspace_elements(capacity, width, heads, kernel)) {}

Error causal_attention(std::span<const float> query, std::span<const float> key,
                       std::span<const float> value, std::size_t tokens, std::span<float> output,
                       AttentionWorkspace &workspace, SequenceOutput mode) noexcept {
    using namespace sequence_detail;
    if (workspace.storage_.size() == 0)
        return Error::invalid_state;
    if ((mode != SequenceOutput::all && mode != SequenceOutput::last) || tokens == 0 ||
        tokens > workspace.capacity_)
        return Error::shape_mismatch;
    const auto width = workspace.width_;
    const auto count = tokens * width;
    const auto first_row = mode == SequenceOutput::last ? tokens - 1 : 0;
    if (query.size() != count || key.size() != count || value.size() != count ||
        output.size() != (tokens - first_row) * width)
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
    for (std::size_t t = first_row; t < tokens; ++t) {
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
            std::size_t c = 0;
            if (workspace.kernel_ == AttentionKernel::tiled) {
                constexpr std::size_t lanes = 8;
                for (; c + lanes <= head_width; c += lanes) {
                    std::array<double, lanes> sums{};
                    // Each channel still visits observations in exactly the scalar order.
                    // Adjacent channels use contiguous values and independent accumulators.
                    for (std::size_t j = 0; j < active; ++j) {
                        const double probability = probabilities[j];
                        const auto *row = value.data() + j * width + channel + c;
                        for (std::size_t lane = 0; lane < lanes; ++lane)
                            sums[lane] += probability * static_cast<double>(row[lane]);
                    }
                    for (std::size_t lane = 0; lane < lanes; ++lane)
                        if (!checked_float(sums[lane], candidate[t * width + channel + c + lane]))
                            return Error::numerical_error;
                }
            }
            // Scalar reference, small heads and incomplete final tiles share this path.
            for (; c < head_width; ++c) {
                double sum = 0;
                for (std::size_t j = 0; j < active; ++j)
                    sum += static_cast<double>(probabilities[j]) * value[j * width + channel + c];
                if (!checked_float(sum, candidate[t * width + channel + c]))
                    return Error::numerical_error;
            }
        }
    }
    std::copy(candidate.begin() + static_cast<std::ptrdiff_t>(first_row * width), candidate.end(),
              output.begin());
    return Error::ok;
}
} // namespace stealth::numeric
